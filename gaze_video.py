# gaze_video.py  –  GAZE TRACKING + JSON REPORT
# python gaze_video.py --video "video.mp4"
# python gaze_video.py --video "video.mp4" --flip --no-emotion --speed 2.0 --output out.mp4 --report out.json

import os
import time
import json
import argparse
import collections

import cv2
import mediapipe as mp
import numpy as np
import gaze_core as gc


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive parameter engine
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveParams:
    UPDATE_EVERY = 45

    def __init__(self):
        self._size_buf = collections.deque(maxlen=90)
        self._cy_buf   = collections.deque(maxlen=90)
        self._yaw_buf  = collections.deque(maxlen=90)
        self._cnt      = 0
        self.ear_th       = gc.EAR_BLINK_TH
        self.conf_min     = gc.CONF_MIN
        self.edge_margin  = gc.EDGE_MARGIN
        self.yaw_scale    = gc.HEAD_YAW_SCALE
        self.pitch_scale  = gc.HEAD_PITCH_SCALE
        self.yaw_offscr   = gc.HEAD_YAW_OFFSCREEN
        self.pitch_offscr = gc.HEAD_PITCH_OFFSCREEN
        self.kf_max_jump  = 0.22
        self.kf_max_vel   = 0.10

    def feed(self, face, hp):
        if face is None:
            return
        self._size_buf.append(gc.face_size_score(face))
        try:
            self._cy_buf.append(float(np.mean([lm.y for lm in face])))
        except Exception:
            pass
        if hp is not None:
            self._yaw_buf.append(abs(hp[1]))
        self._cnt += 1
        if self._cnt % self.UPDATE_EVERY == 0:
            self._recalc()

    def _recalc(self):
        if len(self._size_buf) < 20:
            return

        dist    = float(np.clip(np.median(self._size_buf), 0.25, 1.0))
        med_cy  = float(np.median(self._cy_buf))  if self._cy_buf  else 0.5
        med_yaw = float(np.median(self._yaw_buf)) if self._yaw_buf else 0.0

        self.ear_th      = gc.EAR_BLINK_TH * (1.0 - 0.15 * (1.0 - dist))
        self.conf_min    = float(np.clip(gc.CONF_MIN * dist, 0.15, 0.40))
        self.edge_margin = float(np.clip(gc.EDGE_MARGIN + np.clip(med_yaw / 15, 0, 1) * 0.04, 0.05, 0.18))
        mult = 1.0 + 0.5 * (1.0 - dist)
        self.yaw_scale   = gc.HEAD_YAW_SCALE * mult
        self.pitch_scale = gc.HEAD_PITCH_SCALE * mult
        self.yaw_offscr  = gc.HEAD_YAW_OFFSCREEN * (0.85 + 0.30 * dist)

        if med_cy > 0.58:
            self.pitch_offscr = gc.HEAD_PITCH_OFFSCREEN * 1.35
        elif med_cy < 0.42:
            self.pitch_offscr = gc.HEAD_PITCH_OFFSCREEN * 0.80
        else:
            self.pitch_offscr = gc.HEAD_PITCH_OFFSCREEN * (0.85 + 0.30 * dist)

        self.kf_max_jump = float(np.clip(0.18 + 0.12 * (1 - dist), 0.18, 0.35))
        self.kf_max_vel  = float(np.clip(0.08 + 0.08 * (1 - dist), 0.08, 0.20))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for JSON report
# ─────────────────────────────────────────────────────────────────────────────

def _json_safe_float(x, default=0.0):
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _new_stats():
    return {
        "total_frames": 0,
        "face_frames": 0,
        "no_face_frames": 0,
        "offscreen_frames": 0,
        "low_conf_frames": 0,
        "direction_counts": {
            "LEFT": 0,
            "RIGHT": 0,
            "UP": 0,
            "DOWN": 0,
            "CENTER": 0,
            "UP-LEFT": 0,
            "UP-RIGHT": 0,
            "DOWN-LEFT": 0,
            "DOWN-RIGHT": 0,
            "LOW_CONF": 0,
            "NO_FACE": 0,
        },
        "emotion_counts": {},
        "events": [],
        "frames": [],
    }


def _push_event(events, event_type, start_t, end_t, extra=None):
    item = {
        "type": str(event_type),
        "start_sec": round(float(start_t), 3),
        "end_sec": round(float(end_t), 3),
        "duration_sec": round(float(max(0.0, end_t - start_t)), 3),
    }
    if extra:
        item.update(extra)
    events.append(item)


def _finalize_run_events(stats, current_run, total_duration):
    if current_run is not None:
        _push_event(
            stats["events"],
            event_type=current_run["type"],
            start_t=current_run["start_sec"],
            end_t=total_duration,
            extra=current_run.get("extra", {})
        )


# ─────────────────────────────────────────────────────────────────────────────
# Head-pose dominant yön tahmini
# DÜZELTME: Kafa yatay döndüğünde (|yaw| > eşik) pitch okuması güvenilmez.
# Bu durumda dikey bileşen bastırılır → yanlış "UP/DOWN" önlenir.
# ─────────────────────────────────────────────────────────────────────────────

def direction_from_head(hp, head_center,
                        yaw_th=10.0, pitch_th=8.0) -> tuple[str, bool]:
    if hp is None:
        return "LOW_CONF", False

    cy = head_center[0] if head_center else 0.0
    cp = head_center[1] if head_center else 0.0
    ry = hp[1] - cy    # yaw  sapması  (+ = sağa)
    rp = hp[0] - cp    # pitch sapması (+ = yukarı)

    h = 0
    if ry > yaw_th:
        h = 1
    elif ry < -yaw_th:
        h = -1

    yaw_dominant = abs(ry) > yaw_th * 1.2
    v = 0
    if not yaw_dominant:
        if rp > pitch_th:
            v = -1   # yukarı
        elif rp < -pitch_th:
            v = 1    # aşağı

    is_off = abs(ry) > gc.HEAD_YAW_OFFSCREEN * 0.8 or \
             abs(rp) > gc.HEAD_PITCH_OFFSCREEN * 0.8

    return gc.DIR_TABLE.get((h, v), "CENTER"), is_off


def _head_offscreen(hp, head_center, ap: AdaptiveParams) -> bool:
    if hp is None:
        return False
    cy = head_center[0] if head_center else hp[1]
    cp = head_center[1] if head_center else hp[0]
    ry = hp[1] - cy
    rp = hp[0] - cp
    return (ry / ap.yaw_offscr) ** 2 + (rp / ap.pitch_offscr) ** 2 > 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def video_mode(video_path: str, em_model: str = None,
               no_emotion: bool = False, speed: float = 1.0,
               output_path: str = None, flip: bool = False,
               report_path: str = None):

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video bulunamadı: {video_path}")

    gc.ensure_model()
    detector = gc.build_detector()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {video_path}")

    vid_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_ms = int(1000.0 / max(vid_fps * speed, 1.0))
    print(f"[video] {vid_w}x{vid_h} @ {vid_fps:.1f}fps | {total_fr}fr | x{speed} | flip={flip}")

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, vid_fps, (gc.INFER_W, gc.INFER_H))

    light    = gc.LightAdaptor()
    kf       = gc.KF2DAdaptive(fps=vid_fps)
    mapper   = gc.AffineMapper()
    calib_ok = mapper.load_single()
    if not calib_ok:
        print("[video] UYARI: calib_data.npz yok – head-pose modu aktif.")

    head   = gc.HeadCenterTracker(init_frames=90, alpha=0.005,
                                  min_conf=0.40, iris_center_th=0.12)
    fusion = gc.HeadEyeFusion(head)
    voter  = gc.DirectionVoter(window=8)
    offmon = gc.OffscreenMonitor(hold=gc.OFFSCREEN_HOLD)
    ap     = AdaptiveParams()

    if calib_ok:
        fusion.reset_to(0.5, 0.5)
        voter.seed("CENTER")

    em_tracker  = gc.EmotionTracker(em_model or gc.EM_MODEL_NAME) if not no_emotion else None
    em_dominant = "neutral"
    em_cnt      = 0

    fps = 0.0
    t_fps = time.time()
    fcount = 0
    win = "GAZE TRACKING"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    paused = False

    stats = _new_stats()
    current_dir_run = None
    current_offscreen_run = None
    current_no_face_run = None

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    print("[video] Bitti.")
                    break

                if flip:
                    frame = cv2.flip(frame, 1)

                frame = cv2.resize(frame, (gc.INFER_W, gc.INFER_H), interpolation=cv2.INTER_AREA)
                now   = time.time()
                fcount += 1
                if fcount % 10 == 0:
                    fps = 10.0 / max(now - t_fps, 1e-6)
                    t_fps = now

                frame     = light.process(frame)
                light_str = light.status()

                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms  = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                result = detector.detect_for_video(mp_img, ts_ms)
                face_ok = bool(result.face_landmarks)

                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                t_sec = ts_ms / 1000.0 if ts_ms is not None else (frame_idx / max(vid_fps, 1e-6))
                stats["total_frames"] += 1

                stable_dir = "NO_FACE"
                conf = 0.0
                gp = None
                fusion_w = 0.0
                off_active = False

                if not face_ok:
                    stats["no_face_frames"] += 1
                    stats["direction_counts"]["NO_FACE"] += 1

                    if current_no_face_run is None:
                        current_no_face_run = {
                            "type": "NO_FACE",
                            "start_sec": t_sec,
                            "extra": {}
                        }

                    if current_dir_run is not None:
                        _push_event(
                            stats["events"],
                            event_type=current_dir_run["type"],
                            start_t=current_dir_run["start_sec"],
                            end_t=t_sec,
                            extra=current_dir_run.get("extra", {})
                        )
                        current_dir_run = None

                    if current_offscreen_run is not None:
                        _push_event(
                            stats["events"],
                            event_type=current_offscreen_run["type"],
                            start_t=current_offscreen_run["start_sec"],
                            end_t=t_sec,
                            extra=current_offscreen_run.get("extra", {})
                        )
                        current_offscreen_run = None

                    stats["frames"].append({
                        "frame_idx": frame_idx,
                        "time_sec": round(float(t_sec), 3),
                        "face_ok": False,
                        "direction": "NO_FACE",
                        "offscreen": False,
                        "confidence": 0.0,
                        "emotion": em_dominant,
                        "gaze_pitch": 0.0,
                        "gaze_yaw": 0.0,
                        "gaze_roll": 0.0,
                    })

                    _draw_hud_video(frame, stable_dir, conf, gp, fps,
                                    False, 0.0, light_str,
                                    em_dominant, em_tracker)
                else:
                    stats["face_frames"] += 1

                    if current_no_face_run is not None:
                        _push_event(
                            stats["events"],
                            event_type=current_no_face_run["type"],
                            start_t=current_no_face_run["start_sec"],
                            end_t=t_sec,
                            extra=current_no_face_run.get("extra", {})
                        )
                        current_no_face_run = None

                    face = result.face_landmarks[0]
                    hp   = gc.head_pose_yaw_pitch(result)
                    ap.feed(face, hp)

                    if em_tracker:
                        em_cnt += 1
                        if em_cnt % gc.EM_FEED_EVERY == 0:
                            em_tracker.feed(gc.extract_face_crop(frame, face, align=True))
                        _, em_dominant = em_tracker.tick()

                    ear_l = gc.compute_ear_6pt(face, gc.L_EAR_PTS)
                    ear_r = gc.compute_ear_6pt(face, gc.R_EAR_PTS)
                    blink_bad = (0.5 * (ear_l + ear_r) < ap.ear_th)

                    iris_l = gc.iris_in_eyebox(face, gc.L_EYE, gc.L_IRIS_IDX)
                    iris_r = gc.iris_in_eyebox(face, gc.R_EYE, gc.R_IRIS_IDX)
                    iris_xy = ((iris_l[0] + iris_r[0]) * 0.5, (iris_l[1] + iris_r[1]) * 0.5) \
                              if iris_l and iris_r else (iris_l or iris_r)

                    head_center = head.center()
                    conf = gc.confidence_score(face, ear_l, ear_r, hp, head_center,
                                               frame_quality=light.frame_quality)
                    gp   = gc.gaze_pose_degrees(face, iris_l=iris_l, iris_r=iris_r)

                    kf.max_jump = ap.kf_max_jump
                    kf.max_vel  = ap.kf_max_vel
                    x, y, _ = kf.step(iris_xy, valid=(not blink_bad and iris_xy is not None))
                    sm_iris = (x, y) if iris_xy is not None else None

                    head.update(hp, iris_xy=sm_iris, conf=conf, blink_bad=blink_bad)
                    head_center = head.center()

                    if mapper.ready and sm_iris and conf >= ap.conf_min:
                        gaze_raw = mapper.map(sm_iris, clip=False)
                        gc.HEAD_YAW_SCALE   = ap.yaw_scale
                        gc.HEAD_PITCH_SCALE = ap.pitch_scale
                        fused_raw, fusion_w = fusion.fuse(gaze_raw, hp)
                        direction, is_off = gc.zone_from_xy(
                            fused_raw[0], fused_raw[1],
                            mapper.th_h, mapper.th_v,
                            edge_margin=ap.edge_margin
                        )
                        stable_dir = voter.push(direction)
                        off_active, just_returned = offmon.update(is_off, stable_dir, conf)
                    else:
                        direction, is_off = direction_from_head(hp, head_center)
                        stable_dir = voter.push(direction)
                        off_active, just_returned = offmon.update(is_off, stable_dir, conf)

                    if just_returned:
                        kf.reset()
                        fusion.reset_to(0.5, 0.5)
                        voter.seed("CENTER")

                    _draw_hud_video(frame, stable_dir, conf, gp, fps,
                                    off_active, offmon.dur if off_active else 0.0,
                                    light_str, em_dominant, em_tracker)

                    if stable_dir not in stats["direction_counts"]:
                        stats["direction_counts"][stable_dir] = 0
                    stats["direction_counts"][stable_dir] += 1

                    if stable_dir == "LOW_CONF":
                        stats["low_conf_frames"] += 1

                    if off_active:
                        stats["offscreen_frames"] += 1
                        if current_offscreen_run is None:
                            current_offscreen_run = {
                                "type": "OFFSCREEN",
                                "start_sec": t_sec,
                                "extra": {"direction_at_start": stable_dir}
                            }
                    else:
                        if current_offscreen_run is not None:
                            _push_event(
                                stats["events"],
                                event_type=current_offscreen_run["type"],
                                start_t=current_offscreen_run["start_sec"],
                                end_t=t_sec,
                                extra=current_offscreen_run.get("extra", {})
                            )
                            current_offscreen_run = None

                    if stable_dir not in ("NO_FACE", "LOW_CONF"):
                        if current_dir_run is None or current_dir_run["type"] != stable_dir:
                            if current_dir_run is not None:
                                _push_event(
                                    stats["events"],
                                    event_type=current_dir_run["type"],
                                    start_t=current_dir_run["start_sec"],
                                    end_t=t_sec,
                                    extra=current_dir_run.get("extra", {})
                                )
                            current_dir_run = {
                                "type": stable_dir,
                                "start_sec": t_sec,
                                "extra": {}
                            }
                    else:
                        if current_dir_run is not None:
                            _push_event(
                                stats["events"],
                                event_type=current_dir_run["type"],
                                start_t=current_dir_run["start_sec"],
                                end_t=t_sec,
                                extra=current_dir_run.get("extra", {})
                            )
                            current_dir_run = None

                    if em_dominant:
                        stats["emotion_counts"][em_dominant] = stats["emotion_counts"].get(em_dominant, 0) + 1

                    stats["frames"].append({
                        "frame_idx": frame_idx,
                        "time_sec": round(float(t_sec), 3),
                        "face_ok": True,
                        "direction": stable_dir,
                        "offscreen": bool(off_active),
                        "confidence": round(_json_safe_float(conf), 4),
                        "emotion": em_dominant,
                        "gaze_pitch": round(_json_safe_float(gp[0] if gp else None), 3),
                        "gaze_yaw": round(_json_safe_float(gp[1] if gp else None), 3),
                        "gaze_roll": round(_json_safe_float(gp[2] if gp else None), 3),
                    })

                _draw_progress(frame, fcount, total_fr)
                if writer:
                    writer.write(frame)

            cv2.imshow(win, frame)
            key = cv2.waitKey(1 if paused else frame_ms) & 0xFF

            if key == 27:
                break
            elif key == 32:
                paused = not paused
            elif key in (ord('f'), ord('F')):
                flip = not flip
            elif key in (ord('r'), ord('R')):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                kf.reset()
                fusion.reset()
                voter.reset()
                head.reset()
                offmon.force_close()
                ap.__init__()
                fcount = 0
                fps = 0.0
                t_fps = time.time()
                if em_tracker:
                    em_tracker.reset()
                paused = False

                stats = _new_stats()
                current_dir_run = None
                current_offscreen_run = None
                current_no_face_run = None

    finally:
        total_duration = total_fr / max(vid_fps, 1e-6) if total_fr > 0 else (
            stats["frames"][-1]["time_sec"] if stats["frames"] else 0.0
        )

        _finalize_run_events(stats, current_dir_run, total_duration)
        _finalize_run_events(stats, current_offscreen_run, total_duration)
        _finalize_run_events(stats, current_no_face_run, total_duration)

        summary = {
            "video_path": video_path,
            "output_path": output_path,
            "report_path": report_path,
            "fps": round(float(vid_fps), 3),
            "frame_count": int(total_fr),
            "duration_sec": round(float(total_duration), 3),
            "frames_processed": int(stats["total_frames"]),
            "face_frames": int(stats["face_frames"]),
            "no_face_frames": int(stats["no_face_frames"]),
            "offscreen_frames": int(stats["offscreen_frames"]),
            "low_conf_frames": int(stats["low_conf_frames"]),
            "direction_counts": stats["direction_counts"],
            "emotion_counts": stats["emotion_counts"],
            "events_count": len(stats["events"]),
        }

        report = {
            "summary": summary,
            "events": stats["events"],
            "frames": stats["frames"],
        }

        if report_path:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"[video] JSON rapor yazıldı: {report_path}")

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        try:
            detector.close()
        except Exception:
            pass
        try:
            offmon.force_close()
        except Exception:
            pass
        if em_tracker:
            try:
                em_tracker.stop()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# HUD
# ─────────────────────────────────────────────────────────────────────────────

def _draw_hud_video(frame, stable, conf, gp, fps,
                    off_active, off_dur, light_str,
                    emotion_label, em_tracker):
    H, W = frame.shape[:2]
    col     = gc.DIR_COLOR.get(stable, (180, 180, 180))
    top_col = (0, 0, 50) if not off_active else (70, 0, 0)
    cv2.rectangle(frame, (0, 0), (W, 92), top_col, -1)

    lbl = f"OFF-SCREEN ({off_dur:.1f}s)" if off_active else f"{stable} [{conf:.0%}]"
    (tw, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)
    cv2.putText(frame, lbl, ((W - tw) // 2, 84), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 0), 4)
    cv2.putText(frame, lbl, ((W - tw) // 2, 84), cv2.FONT_HERSHEY_DUPLEX, 0.75, col, 2)

    cv2.putText(frame, f"FPS:{fps:.0f}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160, 160, 160), 1)

    em_avail = em_tracker is not None and em_tracker.available
    if em_avail and emotion_label:
        em_col = gc.EMOTION_META.get(emotion_label, {}).get("color", (180, 180, 180))
        em_lbl = gc.EMOTION_META.get(emotion_label, {}).get("label", emotion_label).upper()
        cv2.putText(frame, f"EMOTION: {em_lbl}", (8, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, em_col, 2)

    if gp:
        cv2.putText(frame, f"EYE P:{gp[0]:+.0f} Y:{gp[1]:+.0f} R:{gp[2]:+.0f}",
                    (8, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (130, 130, 130), 1)

    cv2.putText(frame, light_str, (W - 230, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 180, 100), 1)

    if off_active:
        for t in range(1, 6):
            cv2.rectangle(frame, (t, t), (W - t, H - t), (0, 0, 200), 1)


def _draw_progress(frame, current: int, total: int):
    if total <= 0:
        return
    H, W = frame.shape[:2]
    bar_h  = 6
    filled = int(W * current / total)
    cv2.rectangle(frame, (0, H - bar_h), (W, H), (40, 40, 40), -1)
    cv2.rectangle(frame, (0, H - bar_h), (filled, H), (0, 200, 120), -1)
    cv2.putText(frame, f"{100 * current / total:.0f}%  {current}/{total}",
                (8, H - bar_h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GAZE TRACKING — Video")
    ap.add_argument("--video",      type=str, required=True)
    ap.add_argument("--output",     type=str, default=None)
    ap.add_argument("--report",     type=str, default=None)
    ap.add_argument("--em-model",   type=str, default=gc.EM_MODEL_NAME)
    ap.add_argument("--no-emotion", action="store_true")
    ap.add_argument("--speed",      type=float, default=1.0)
    ap.add_argument("--flip",       action="store_true",
                    help="Yatay ayna çevir (telefon ön kamera için)")
    args = ap.parse_args()

    gc.EM_MODEL_NAME = args.em_model
    video_mode(args.video, args.em_model, args.no_emotion,
               args.speed, args.output, args.flip, args.report)