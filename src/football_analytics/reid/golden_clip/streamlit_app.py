"""Streamlit Target Failure Window Pilot — static event review (R1.2).

Default path: native st.video + static PNG + streamlit-image-coordinates.
Custom interactive declare_component is OFF by default (Advanced only).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

# Heavy model imports intentionally absent.


def _ensure_src_path() -> None:
    import sys

    src = Path(__file__).resolve().parents[3]
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_ensure_src_path()

from football_analytics.reid.golden_clip import UI_PROFILE  # noqa: E402
from football_analytics.reid.golden_clip.annotation_log import (  # noqa: E402
    AnnotationLog,
    assert_not_product_log,
)
from football_analytics.reid.golden_clip.pilot import (  # noqa: E402
    CANDIDATE_REQUIRED_LABELS,
    COVERAGE_SCOPE,
    NOT_MEASURABLE_FROM_FAILURE_WINDOW_PILOT,
    PILOT_LABELS,
    build_pilot_label_event,
    choose_pilot_next_gate,
    empty_pilot_doc,
    summarize_pilot_labels,
)
from football_analytics.reid.golden_clip.schema import new_session_id  # noqa: E402
from football_analytics.reid.golden_clip.static_packages import (  # noqa: E402
    build_enrollment_package,
    build_static_failure_packages,
    load_package_manifest,
    pick_enrollment_frame,
    resolve_enrollment_click,
)
from football_analytics.reid.golden_clip.window_index import (  # noqa: E402
    load_dense_observations_index,
)
from football_analytics.reid.hil_ui.compat import streamlit_image  # noqa: E402


def _root_from_env() -> Path:
    raw = os.environ.get("GOLDEN_CLIP_ROOT")
    if not raw:
        raise RuntimeError("GOLDEN_CLIP_ROOT env var required")
    return Path(raw).expanduser().resolve()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _perf_log(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _labeled_window_ids(events: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for ev in events:
        if ev.get("action") != "PILOT_FAILURE_WINDOW_LABEL":
            continue
        wid = ev.get("failure_window_id") or (ev.get("pilot") or {}).get(
            "failure_window_id"
        )
        if wid:
            out.add(str(wid))
    return out


def run_app() -> None:
    import streamlit as st
    from PIL import Image
    from streamlit_image_coordinates import streamlit_image_coordinates

    t_page0 = time.perf_counter()
    st.set_page_config(page_title="Target Failure Window Pilot", layout="wide")
    st.title("Target Failure Window Pilot — Static Event Review")
    st.caption(
        f"ui_profile={UI_PROFILE} · coverage_scope={COVERAGE_SCOPE} · "
        "default=static (no custom video component) · product logs unaffected"
    )
    st.info(
        "R1.2: Interactive custom component varsayılan kapalı. "
        "Hedefi statik karede seçin; failure window’ları native clip + PNG ile etiketleyin."
    )

    root = _root_from_env()
    refs = _load_json(root / "read_only_refs" / "artifact_pointers.json")
    session_dir = root / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    static_dir = session_dir / "static_failure_windows"
    static_dir.mkdir(parents=True, exist_ok=True)
    ann_path = session_dir / "annotation_log.jsonl"
    assert_not_product_log(ann_path)
    meta_path = session_dir / "annotation_session.json"
    pilot_path = session_dir / "failure_window_pilot.json"
    perf_path = session_dir / "ui_perf.jsonl"

    @st.cache_data(show_spinner=False)
    def _cached_index(dense_path: str, mtime: float) -> dict[str, Any]:
        return load_dense_observations_index(Path(dense_path))

    if "gt_session" not in st.session_state:
        if meta_path.is_file():
            st.session_state.gt_session = _load_json(meta_path)
        else:
            sess = {
                "annotation_session_id": new_session_id(),
                "reviewer": "golden_clip_reviewer",
                "source_video_sha256": refs["source_video_sha256"],
                "match_id": refs["match_id"],
                "analysis_run_id": refs["analysis_run_id"],
                "target_id": refs["target_id"],
                "fps": refs["fps"],
                "frame_count": refs["frame_count"],
                "mode": "TARGET_FAILURE_WINDOW_PILOT",
                "ui_default": "static",
            }
            _write_json(meta_path, sess)
            st.session_state.gt_session = sess

    for k, default in (
        ("selection", {}),
        ("packages_index", None),
        ("window_idx", 0),
        ("submitted_event_uuids", []),
        ("labeled_window_ids", []),
        ("page_load_ms", None),
        ("selection_latency_ms", None),
        ("nav_latency_ms", None),
        ("submit_latency_ms", None),
        ("enable_experimental_interactive", False),
    ):
        if k not in st.session_state:
            st.session_state[k] = default

    sess = st.session_state.gt_session
    fps = float(sess["fps"])
    frame_count = int(sess["frame_count"])
    log = AnnotationLog(ann_path)

    # Restore labeled ids from log (idempotence across reruns)
    events = log.read_raw()
    st.session_state.labeled_window_ids = sorted(_labeled_window_ids(events))

    dense_path = Path(refs["dense_bbox_timeline"])
    if not dense_path.is_file():
        dense_path = _project_root() / refs["dense_bbox_timeline"]
    idx = _cached_index(str(dense_path), dense_path.stat().st_mtime)
    obs_all = idx["observations_by_frame"]

    video = Path(refs["source_video"])
    if not video.is_file():
        video = _project_root() / refs["source_video"]

    # ---- metrics strip (no full overlay/metrics calc) ----
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("UI mode", "static")
    m2.metric(
        "Selection ms",
        st.session_state.selection_latency_ms
        if st.session_state.selection_latency_ms is not None
        else "—",
    )
    m3.metric(
        "Submit ms",
        st.session_state.submit_latency_ms
        if st.session_state.submit_latency_ms is not None
        else "—",
    )
    m4.metric("Labeled", len(st.session_state.labeled_window_ids))

    # ---- Step 1: static enrollment ----
    enroll_dir = static_dir / "enrollment"
    enroll_man_path = enroll_dir / "enrollment_manifest.json"
    if not enroll_man_path.is_file():
        with st.spinner("Enrollment frame hazırlanıyor (tek PNG)…"):
            t0 = time.perf_counter()
            fi = pick_enrollment_frame(obs_all, frame_count=frame_count)
            enroll = build_enrollment_package(
                source_video=video,
                source_video_sha256=refs["source_video_sha256"],
                observations_by_frame=obs_all,
                frame_index=fi,
                output_dir=enroll_dir,
            )
            _perf_log(
                perf_path,
                {
                    "event": "enrollment_build",
                    "ms": int((time.perf_counter() - t0) * 1000),
                    "frame_index": fi,
                    "candidate_count": enroll["candidate_count"],
                },
            )
    else:
        enroll = _load_json(enroll_man_path)

    st.markdown("### 1) Hedef seçimi (statik kare)")
    st.caption(
        f"Frame {enroll['frame_index']} · tıklanabilir PNG · "
        "custom video component yok · yalnız bu kare adayları"
    )
    if not st.session_state.selection:
        png = Path(enroll["png"]["path"])
        shown = Image.open(png).convert("RGB").resize(
            (int(enroll["display_w"]), int(enroll["display_h"]))
        )
        t_sel0 = time.perf_counter()
        click = streamlit_image_coordinates(
            shown,
            key="gt_r12_enrollment_click_stable",
            width=int(enroll["display_w"]),
        )
        if click and "x" in click and "y" in click:
            hit = resolve_enrollment_click(
                ui_x=float(click["x"]),
                ui_y=float(click["y"]),
                enrollment=enroll,
            )
            if hit and hit.get("raw_track_id"):
                st.session_state.selection_latency_ms = int(
                    (time.perf_counter() - t_sel0) * 1000
                )
                st.session_state.selection = {
                    "selected_raw_track_id": str(hit["raw_track_id"]),
                    "selected_segment_id": hit.get("segment_id"),
                    "selected_frame_index": int(enroll["frame_index"]),
                    "selected_bbox_xyxy": hit.get("bbox_xyxy"),
                    "detection_id": hit.get("detection_id"),
                }
                _perf_log(
                    perf_path,
                    {
                        "event": "target_selection",
                        "ms": st.session_state.selection_latency_ms,
                        "raw_track_id": hit["raw_track_id"],
                    },
                )
                st.rerun()
            else:
                st.caption("Tıklama bir bbox içine düşmedi.")
        # Optional radio fallback if click is awkward
        opts = [
            f"{c['raw_track_id']} | {c.get('segment_id')} | {c.get('bbox_xyxy')}"
            for c in enroll.get("candidates") or []
        ]
        pick = st.selectbox("veya listeden seç", ["(none)"] + opts, key="enroll_radio")
        if pick != "(none)" and st.button("Hedefi seç (liste)", type="primary"):
            c = (enroll.get("candidates") or [])[opts.index(pick)]
            st.session_state.selection = {
                "selected_raw_track_id": str(c["raw_track_id"]),
                "selected_segment_id": c.get("segment_id"),
                "selected_frame_index": int(enroll["frame_index"]),
                "selected_bbox_xyxy": c.get("bbox_xyxy"),
                "detection_id": c.get("detection_id"),
            }
            st.rerun()
        st.stop()

    sel = st.session_state.selection
    st.success(
        f"Hedef seed · raw_track={sel['selected_raw_track_id']} · "
        f"segment={sel.get('selected_segment_id')} · frame={sel.get('selected_frame_index')}"
    )
    if st.button("Hedefi sıfırla"):
        st.session_state.selection = {}
        st.session_state.packages_index = None
        st.session_state.window_idx = 0
        st.rerun()

    # ---- Step 2: build packages once for this track ----
    track_pkg_dir = static_dir / f"track_{sel['selected_raw_track_id']}"
    index_path = track_pkg_dir / "packages_index.json"
    if st.session_state.packages_index is None:
        if index_path.is_file():
            st.session_state.packages_index = _load_json(index_path)
        else:
            with st.spinner(
                "Failure window paketleri üretiliyor (5 küçük MP4/PNG; bir kez)…"
            ):
                t0 = time.perf_counter()
                st.session_state.packages_index = build_static_failure_packages(
                    source_video=video,
                    source_video_sha256=refs["source_video_sha256"],
                    observations_by_frame=obs_all,
                    selected_raw_track_id=str(sel["selected_raw_track_id"]),
                    selected_segment_id=sel.get("selected_segment_id"),
                    fps=fps,
                    frame_count=frame_count,
                    output_dir=track_pkg_dir,
                    max_windows=5,
                )
                _perf_log(
                    perf_path,
                    {
                        "event": "packages_build",
                        "ms": int((time.perf_counter() - t0) * 1000),
                        "package_count": st.session_state.packages_index["package_count"],
                    },
                )

    packages = list((st.session_state.packages_index or {}).get("packages") or [])
    if not packages:
        st.error("Failure window üretilemedi.")
        st.stop()

    # Skip already labeled windows when navigating
    unlabeled = [
        i
        for i, p in enumerate(packages)
        if str(p["event_id"]) not in set(st.session_state.labeled_window_ids)
    ]
    if not unlabeled:
        st.balloons()
        st.success("Tüm candidate window’lar etiketlendi.")
        summary = summarize_pilot_labels(log.read_raw())
        gate = choose_pilot_next_gate(summary)
        st.write(summary)
        st.write(gate)
        st.stop()

    idx_w = int(st.session_state.window_idx)
    if idx_w not in unlabeled:
        idx_w = unlabeled[0]
        st.session_state.window_idx = idx_w
    pkg_sum = packages[idx_w]
    # Load ONLY this window manifest + assets
    man = load_package_manifest(Path(pkg_sum["manifest_path"]))

    st.markdown(
        f"### 2) Failure window {unlabeled.index(idx_w) + 1}/"
        f"{len(unlabeled)} remaining · `{pkg_sum['event_id']}`"
    )
    nav1, nav2, nav3 = st.columns(3)
    if nav1.button("Önceki window") and unlabeled.index(idx_w) > 0:
        t0 = time.perf_counter()
        st.session_state.window_idx = unlabeled[unlabeled.index(idx_w) - 1]
        st.session_state.nav_latency_ms = int((time.perf_counter() - t0) * 1000)
        st.rerun()
    nav2.write(
        f"kind={pkg_sum.get('kind')} · end_f={pkg_sum.get('track_end_frame')} · "
        f"t={pkg_sum.get('track_end_time'):.2f}s"
    )
    if nav3.button("Sonraki window") and unlabeled.index(idx_w) < len(unlabeled) - 1:
        t0 = time.perf_counter()
        st.session_state.window_idx = unlabeled[unlabeled.index(idx_w) + 1]
        st.session_state.nav_latency_ms = int((time.perf_counter() - t0) * 1000)
        st.rerun()

    c_vid, c_img = st.columns(2)
    with c_vid:
        st.markdown("**Native event clip**")
        st.video(str(pkg_sum["clip_path"]))
        st.caption(
            f"previous raw_track={man['previous_raw_track_id']} · "
            f"gap={man.get('temporal_gap_frames')} · "
            f"det_continuity={man.get('detection_continuity_frames')} · "
            f"disp_px={man.get('spatial_displacement_px')} · "
            f"overlap={man.get('bbox_overlap')}"
        )
    with c_img:
        st.markdown("**Candidate bbox frame (static)**")
        streamlit_image(
            st,
            str(pkg_sum["all_candidates_png"]),
            use_column_width=True,
            caption="all candidates on failure frame",
        )

    cand_ids = list(man.get("candidate_raw_track_ids") or [])
    labels = sorted(PILOT_LABELS)

    st.markdown("### 3) Annotation (form — yalnız submit’te yazılır)")
    with st.form(key=f"fw_form_{pkg_sum['event_id']}", clear_on_submit=False):
        label = st.radio("Hata türü", labels, horizontal=False)
        cand = st.radio(
            "Continuation candidate (gerekli: fragment / short occlusion / border)",
            ["(none)"] + cand_ids,
            horizontal=False,
        )
        comment = st.text_input("Reviewer comment", value="")
        submitted = st.form_submit_button("Kaydet ve sonraki olaya geç", type="primary")

    if submitted:
        t0 = time.perf_counter()
        event_uuid = f"euuid_{uuid.uuid4().hex[:16]}"
        # Idempotence: same window already labeled
        if str(pkg_sum["event_id"]) in set(st.session_state.labeled_window_ids):
            st.warning("Bu window zaten etiketlendi (idempotent skip).")
        elif event_uuid in set(st.session_state.submitted_event_uuids):
            st.warning("Duplicate event_uuid (idempotent skip).")
        else:
            try:
                next_tid = None if cand == "(none)" else str(cand)
                if label in CANDIDATE_REQUIRED_LABELS and not next_tid:
                    st.error(f"{label} için candidate zorunlu.")
                else:
                    # Lightweight window dict for event (no PNG/video bytes)
                    window_meta = {
                        "window_id": pkg_sum["event_id"],
                        "event_id": pkg_sum["event_id"],
                        "kind": pkg_sum.get("kind"),
                        "start_frame": man.get("clip", {}).get("start_frame")
                        or man["track_end_frame"],
                        "end_frame": man.get("clip", {}).get("end_frame")
                        or man["track_end_frame"],
                        "start_time": man.get("clip", {}).get("start_time"),
                        "end_time": man.get("clip", {}).get("end_time"),
                        "track_end_frame": man["track_end_frame"],
                        "track_end_time": man["track_end_time"],
                        "previous_raw_track_id": man["previous_raw_track_id"],
                        "possible_next_raw_tracks": man.get("possible_next_raw_tracks"),
                        "time_gap_frames": man.get("temporal_gap_frames"),
                        "detection_continuity_frames": man.get(
                            "detection_continuity_frames"
                        ),
                        "image_position_displacement_px": man.get(
                            "spatial_displacement_px"
                        ),
                        "bbox_overlap": man.get("bbox_overlap"),
                    }
                    ev = build_pilot_label_event(
                        window=window_meta,
                        label=label,
                        reviewer=sess["reviewer"],
                        annotation_session_id=sess["annotation_session_id"],
                        source_video_sha256=sess["source_video_sha256"],
                        match_id=sess["match_id"],
                        analysis_run_id=sess["analysis_run_id"],
                        target_id=sess["target_id"],
                        selected_next_raw_track_id=next_tid,
                        comment=comment,
                        event_uuid=event_uuid,
                        failure_window_id=pkg_sum["event_id"],
                        ui_mode="static",
                    )
                    log.append(ev)
                    uuids = list(st.session_state.submitted_event_uuids or [])
                    uuids.append(event_uuid)
                    st.session_state.submitted_event_uuids = uuids[-200:]
                    labeled = list(st.session_state.labeled_window_ids or [])
                    labeled.append(str(pkg_sum["event_id"]))
                    st.session_state.labeled_window_ids = labeled

                    pilot = (
                        _load_json(pilot_path)
                        if pilot_path.is_file()
                        else empty_pilot_doc(
                            source_video_sha256=sess["source_video_sha256"],
                            match_id=sess["match_id"],
                            analysis_run_id=sess["analysis_run_id"],
                            target_id=sess["target_id"],
                            annotation_session_id=sess["annotation_session_id"],
                            reviewer=sess["reviewer"],
                        )
                    )
                    pilot.setdefault("windows", []).append(
                        {
                            "window_id": pkg_sum["event_id"],
                            "label": label,
                            "event_id": ev["event_id"],
                            "event_uuid": event_uuid,
                            "selected_next_raw_track_id": next_tid,
                            "center_frame": man["track_end_frame"],
                        }
                    )
                    summary = summarize_pilot_labels(
                        [
                            e
                            for e in log.read_raw()
                            if e.get("action") == "PILOT_FAILURE_WINDOW_LABEL"
                        ]
                    )
                    gate = choose_pilot_next_gate(summary)
                    pilot["summary"] = summary
                    pilot["exact_next_gate"] = gate
                    _write_json(pilot_path, pilot)
                    _write_json(session_dir / "pilot_summary.json", summary)
                    _write_json(session_dir / "pilot_next_gate.json", gate)
                    st.session_state.submit_latency_ms = int(
                        (time.perf_counter() - t0) * 1000
                    )
                    _perf_log(
                        perf_path,
                        {
                            "event": "annotation_submit",
                            "ms": st.session_state.submit_latency_ms,
                            "failure_window_id": pkg_sum["event_id"],
                            "label": label,
                        },
                    )
                    # Advance to next unlabeled
                    still = [
                        i
                        for i, p in enumerate(packages)
                        if str(p["event_id"])
                        not in set(st.session_state.labeled_window_ids)
                    ]
                    st.session_state.window_idx = still[0] if still else idx_w
                    st.success(
                        f"Kaydedildi: {label} · labeled={summary['labeled_window_count']}"
                    )
                    st.rerun()
            except Exception as exc:  # noqa: BLE001 — show to reviewer
                st.error(str(exc))

    with st.expander("Honesty / metrics (NOT measurable from pilot)"):
        summary = summarize_pilot_labels(log.read_raw())
        st.write(summary)
        st.write(choose_pilot_next_gate(summary))
        st.write(
            {
                "coverage_scope": COVERAGE_SCOPE,
                "full_metrics": NOT_MEASURABLE_FROM_FAILURE_WINDOW_PILOT,
                "custom_interactive_component_default": False,
                "one_window_assets_only": True,
                "product_logs_path_refused": True,
            }
        )

    # ---- Advanced: experimental interactive (OFF by default) ----
    with st.expander("Advanced / Experimental Interactive Mode", expanded=False):
        st.warning(
            "R1.1’de kullanıcı ortamında donma görüldü. "
            "Pilot başarı kriteri bu moda bağlı değildir. Varsayılan kapalı."
        )
        enable = st.checkbox(
            "Enable experimental interactive component",
            value=False,
            key="exp_interactive_toggle",
        )
        st.session_state.enable_experimental_interactive = bool(enable)
        if enable:
            st.caption(
                "Component yalnız bu expander açık ve checkbox işaretliyken yüklenir. "
                "Full dense manifest gönderilmez."
            )
            # Lazy import — never at page top / default path
            from football_analytics.reid.golden_clip.window_index import (  # noqa: WPS433
                DEFAULT_WINDOW_RADIUS_FRAMES,
                slice_window,
            )
            from football_analytics.reid.hil_ui.interactive_video_component import (  # noqa: WPS433
                interactive_video_review,
            )
            from football_analytics.reid.hil_ui.interactive_video_media import (  # noqa: WPS433
                ensure_interactive_review_proxy,
            )

            center = int(man["track_end_frame"])
            window_obs = slice_window(
                obs_all,
                center_frame=center,
                radius=DEFAULT_WINDOW_RADIUS_FRAMES,
                frame_count=frame_count,
            )
            proxy_path = session_dir / "interactive_review" / "source_proxy_960.mp4"
            proxy = ensure_interactive_review_proxy(
                source_video=video,
                source_video_sha256=refs["source_video_sha256"],
                output_path=proxy_path,
            )
            interactive_video_review(
                video_data_url=proxy["data_url"],
                observations_by_frame=window_obs,
                markers=[],
                selected_raw_track_id=sel.get("selected_raw_track_id"),
                selected_segment_id=sel.get("selected_segment_id"),
                fps=fps,
                video_width=int(refs["resolution"]["width"]),
                video_height=int(refs["resolution"]["height"]),
                key="gt_r12_experimental_interactive_off_by_default",
            )

    if st.session_state.page_load_ms is None:
        st.session_state.page_load_ms = int((time.perf_counter() - t_page0) * 1000)
        _perf_log(
            perf_path,
            {"event": "page_initial_load", "ms": st.session_state.page_load_ms},
        )


if __name__ == "__main__":
    run_app()
