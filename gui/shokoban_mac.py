"""書庫番の実用画面（実用版）.

起動:
    SHOKOBAN_WORK_DIR=~/shokoban-data \\
    SHOKOBAN_DATA_MODE=real \\
    streamlit run gui/shokoban_mac.py

``SHOKOBAN_DATA_MODE`` は real（本番データ）か synthetic（架空データ）が必須。
real は ``<WORK>/store/``、synthetic は ``<WORK>/store-synthetic/`` を使う。
架空顧客の種まき（seed）はしない。

画面は5つ:「案件一覧」「案件ファイル」「名前の確認」「送る前の確認」「手元AI」。
「名前の確認」「送る前の確認」は、本番データ（real）では ``SHOKOBAN_REVIEW_REAL=1`` のときだけ開く。

外から来た文字（手元AIの答え・お客様名・題・見出し・根拠・使った記録）は、
Markdown として解釈させない。本文は ``st.text``（``st.code``・``st.text_area``）で
そのまま出し、ラベルに入れざるを得ない時だけ ``safe_text.escape_label`` を通す。
``st.markdown``・``st.write``・``st.info/success/warning/error`` には固定の文言だけ。
"""

from __future__ import annotations

import html
import os
from datetime import date, datetime
from pathlib import Path

import streamlit as st

from shokoban import llm_runtime
from shokoban.case_chat import CaseChatError, answer_case_question
from shokoban.customer_store import (
    CLEAN_ROOT_CONFIRMATION,
    CustomerStore,
    CustomerStoreError,
    RecordValidationError,
    RevisionConflictError,
)
from shokoban.inbox_import import (
    ARCHIVE_DIR_NAME,
    INBOX_FILE_PATTERN,
    JST,
    STATUS_FAILED,
    STATUS_IMPORTED,
    STATUS_SKIPPED,
    InboxImportError,
    import_inbox,
)
from shokoban.llm_runtime import LLMRuntimeError, load_runtime_config
from shokoban.local_llm import LocalLLMError, generate_local_text
from shokoban.mac_app import (
    DATA_MODE_ENV,
    DATA_MODES,
    WORK_DIR_ENV,
    MacAppError,
    MacAppPaths,
    ensure_inbox,
    list_draft_meetings,
    load_share_labels,
    load_status_record,
    mac_app_paths_from_env,
    prepare_local_llm,
    runtime_status,
    save_share_label,
    save_status_record,
    share_dir_for_mode,
    status_local_dir_for_mode,
    status_record_path,
    store_root_for_mode,
    synthetic_inbox_filter,
)
from shokoban.mask_r3 import MaskR3Error, dry_tagger, make_llama_tagger, tagger_model
from shokoban.name_dict import NameDictError
from shokoban.send_review import (
    HideWord,
    SendReviewError,
    check_hide_words,
    save_hide_words,
    add_ok_phrases,
    load_ok_phrases,
    ok_phrases_path,
    approval_path,
    check_payload_approval,
    final_masked,
    final_texts,
    load_approval,
    payload_digest,
    prepare_send_review,
    save_approval,
)
from shokoban.cloud_memo import (
    CloudMemoError,
    cloud_reader_from_env,
    load_memo,
    make_memo,
    memo_path,
    save_edited,
    save_memo,
    load_usage,
    monthly_limit,
    reserve_reader_call,
    usage_path,
    READER_FAKE,
    reader_is_local,
    share_memo,
    share_memo_path,
    split_memo_sections,
)
from shokoban.assemble_r3 import F_PATH, F_UNDECIDED, speaker_mark
from shokoban.name_review import (
    CHOICE_LABELS,
    JUDGMENT_ORDER,
    NameReviewError,
    build_review,
    collect_decisions,
    count_choices,
    highlight_html,
    common_dict_path,
    load_review_dicts,
    meeting_sentences,
    name_dict_path,
    require_review_mode,
    review_real_enabled,
    save_decisions,
)
from shokoban.progress_status import (
    FIELDS,
    MOVER,
    STAGES,
    TASKS,
    ProgressStatus,
    ProgressStatusError,
    infer_progress_status,
    validate_share_label,
    write_share_file,
)
from shokoban.safe_text import escape_label

MODE_LABELS = {"real": "本番データ", "synthetic": "架空データ"}
SCREENS = (
    ("cases", "案件一覧"),
    ("customer", "案件ファイル"),
    ("names", "名前の確認"),
    ("send", "送る前の確認"),
    ("llm", "手元AI"),
)
IMPORT_STATUS_LABELS = {
    STATUS_IMPORTED: "取り込み",
    STATUS_SKIPPED: "見送り",
    STATUS_FAILED: "失敗",
}

st.set_page_config(page_title="書庫番", page_icon="📚", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1120px;
        padding-top: 2rem;
    }
    div[data-testid="stButton"] > button {
        min-height: 2.8rem;
        border-radius: 0.35rem;
        font-weight: 650;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(128, 128, 128, 0.25);
    }
    @media (max-width: 600px) {
        .block-container {
            padding: 1rem 0.8rem;
        }
        div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap;
            gap: 0.35rem;
        }
        div[data-testid="column"] {
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def jst_now() -> datetime:
    """JST の今（取り込み・メモ・進行の状態の日付に使う）."""
    return datetime.now(JST)


def current_mode() -> str:
    """``SHOKOBAN_DATA_MODE`` を読み、real / synthetic 以外なら画面を止める."""
    raw = os.environ.get(DATA_MODE_ENV)
    if raw is None or not str(raw).strip():
        st.error(
            f"{DATA_MODE_ENV} がありません。real か synthetic を指定してください。"
        )
        st.stop()
    mode = str(raw).strip()
    if mode not in DATA_MODES:
        st.error(f"{DATA_MODE_ENV} は real か synthetic を指定してください。")
        st.stop()
    return mode


def mac_paths_or_stop() -> MacAppPaths:
    """``SHOKOBAN_WORK_DIR`` から保存先を決める（無ければ画面を止める）."""
    try:
        paths = mac_app_paths_from_env()
    except MacAppError as exc:
        st.error(str(exc))
        st.stop()
        raise
    if paths is None:
        st.error(
            f"{WORK_DIR_ENV} がありません。作業フォルダを指定してください。"
        )
        st.stop()
        raise
    return paths


def open_store(paths: MacAppPaths, mode: str) -> CustomerStore:
    """モードに応じた保存先を用意する（初回だけ initialize。種まきはしない）."""
    store = CustomerStore(store_root_for_mode(paths, mode))
    try:
        store.validate_schema_manifest()
    except Exception:
        try:
            store.initialize(confirmation=CLEAN_ROOT_CONFIRMATION)
        except (CustomerStoreError, RecordValidationError) as exc:
            st.error(
                "保存先の顧客ストアを用意できませんでした: "
                + escape_label(str(exc))
            )
            st.stop()
    return store


def inbox_target_count(inbox: Path, *, mode: str) -> int:
    """受け取りフォルダの対象ファイル数（synthetic のときだけ名前を絞る）."""
    if inbox.is_symlink() or not inbox.is_dir():
        return 0
    try:
        entries = sorted(inbox.iterdir(), key=lambda path: path.name)
    except OSError:
        return 0
    count = 0
    for entry in entries:
        if entry.name == ARCHIVE_DIR_NAME:
            continue
        if entry.is_symlink() or not entry.is_file():
            continue
        if entry.name.startswith(".") or entry.name.endswith(".part"):
            continue
        matched = INBOX_FILE_PATTERN.fullmatch(entry.name)
        if matched is None:
            continue
        name_part = matched.group(2).strip()
        if mode == "synthetic" and not synthetic_inbox_filter(name_part):
            continue
        count += 1
    return count


def open_customer(customer_id: str) -> None:
    """案件ファイル画面へ移る."""
    st.session_state.shokoban_customer_id = customer_id
    st.session_state.shokoban_screen = "customer"
    st.rerun()


def render_nav() -> None:
    columns = st.columns(len(SCREENS))
    for column, (key, label) in zip(columns, SCREENS):
        selected = st.session_state.shokoban_screen == key
        if column.button(
            label,
            key=f"nav_{key}",
            type="primary" if selected else "secondary",
        ):
            st.session_state.shokoban_screen = key
            st.rerun()


def render_cases(store: CustomerStore, paths: MacAppPaths, mode: str) -> None:
    st.header("案件一覧")

    st.subheader("受け取りフォルダ")
    st.caption(
        "下のフォルダに、文字起こしを「2026-01-15 お客様名.txt」の形の名前（日付・半角スペース・"
        "お客様名。.txt か .md、UTF-8）で置いてから取り込みます。取り込んだファイルは"
        "「取り込み済み」へ移ります。"
    )
    st.code(str(paths.inbox), language=None)
    if mode == "synthetic":
        st.caption("架空データのモードでは、お客様名が「架空顧客」で始まるファイルだけを取り込みます。")
    st.write(f"対象ファイル: {inbox_target_count(paths.inbox, mode=mode)}件")
    if st.button("受け取りフォルダを取り込む", key="run_inbox_import"):
        try:
            results = import_inbox(
                store,
                paths.inbox,
                now=jst_now(),
                accept_name=synthetic_inbox_filter if mode == "synthetic" else None,
            )
        except (InboxImportError, CustomerStoreError) as exc:
            st.error(
                "取り込みを実行できませんでした: " + escape_label(str(exc))
            )
        else:
            st.session_state.shokoban_import_results = [
                {
                    "ファイル名": result.source_name,
                    "結果": IMPORT_STATUS_LABELS.get(result.status, result.status),
                    "理由": result.reason,
                }
                for result in results
            ]

    import_results = st.session_state.get("shokoban_import_results")
    if import_results:
        st.dataframe(import_results, hide_index=True)
    elif "shokoban_import_results" in st.session_state:
        st.info("取り込み対象のファイルはありません。")

    st.subheader("お客様")
    customers = store.list_customers()
    if not customers:
        st.info("まだお客様がいません。")
    else:
        header = st.columns([3, 2, 1, 2, 1])
        for column, label in zip(
            header,
            ("名前", "確定済みの打ち合わせ数", "下書き数", "最後の打ち合わせ日", ""),
        ):
            column.markdown(f"**{label}**")
        for customer in customers:
            meetings = store.list_meetings(customer["customer_id"])
            confirmed = [
                meeting
                for meeting in meetings
                if meeting["meeting_status"] == "confirmed"
            ]
            drafts = [
                meeting for meeting in meetings if meeting["meeting_status"] == "draft"
            ]
            last_day = confirmed[0]["occurred_at"][:10] if confirmed else "—"
            row = st.columns([3, 2, 1, 2, 1])
            row[0].text(customer["display_name"])
            row[1].write(len(confirmed))
            row[2].write(len(drafts))
            row[3].write(last_day)
            if row[4].button(
                "開く", key=f"open_customer_{customer['customer_id']}"
            ):
                open_customer(customer["customer_id"])

    st.subheader("新しいお客様を手で作る")
    new_name = st.text_input("名前", key="new_customer_name")
    if st.button("お客様を作る", key="create_customer"):
        name = new_name.strip()
        if not name:
            st.error("名前を入れてください。")
        else:
            try:
                customer_id = store.resolve_customer(name)
                if customer_id is None:
                    customer_id = store.create_customer(display_name=name)[
                        "customer_id"
                    ]
            except CustomerStoreError:
                st.error("お客様を作れませんでした。")
            else:
                open_customer(customer_id)


def render_drafts(store: CustomerStore, customer_id: str) -> None:
    drafts = [
        meeting
        for meeting in store.list_meetings(customer_id)
        if meeting["meeting_status"] == "draft"
    ]
    st.subheader("下書き")
    if not drafts:
        st.info("確定待ちの下書きはありません。")
    for draft in drafts:
        st.subheader(
            f"{draft['occurred_at'][:10]} ・ {escape_label(draft['title'])}"
        )
        edited = st.text_area(
            "本文",
            value=draft["details"],
            height=240,
            key=f"draft_details_{draft['record_id']}",
        )
        checked = st.checkbox(
            "内容を確認しました",
            key=f"draft_checked_{draft['record_id']}",
        )
        if st.button(
            "確定する",
            disabled=not checked,
            type="primary",
            key=f"confirm_draft_{draft['record_id']}",
        ):
            try:
                revision = draft["revision"]
                if edited != draft["details"]:
                    updated = store.update_meeting_draft(
                        customer_id,
                        draft["record_id"],
                        expected_revision=revision,
                        title=draft["title"],
                        summary=draft["summary"],
                        details=edited,
                        decisions=list(draft["decisions"]),
                        next_actions=list(draft["next_actions"]),
                    )
                    revision = updated["revision"]
                store.confirm_meeting(
                    customer_id,
                    draft["record_id"],
                    expected_revision=revision,
                    confirmed_by="gui",
                )
            except RevisionConflictError:
                st.error("他で変更されました。再読込してください。")
            except CustomerStoreError:
                st.error("確定できませんでした。再読込してください。")
            else:
                st.session_state.shokoban_notice = "下書きを確定しました。"
                st.rerun()


def render_records(store: CustomerStore, customer_id: str) -> None:
    confirmed = [
        meeting
        for meeting in store.list_meetings(customer_id)
        if meeting["meeting_status"] == "confirmed"
    ]
    st.subheader("打ち合わせ記録")
    if not confirmed:
        st.info("確定済みの打ち合わせはまだありません。")
    for meeting in confirmed:
        heading = (
            f"{meeting['occurred_at'][:10]} ・ {escape_label(meeting['title'])}"
        )
        with st.expander(heading):
            if meeting["summary"]:
                st.text(meeting["summary"])
            st.text(meeting["details"])
            # 確定のあとで直したいとき: 下書きに戻し、上の「下書き」で直して、もう一度確定する
            reopen_checked = st.checkbox(
                "下書きに戻して直す（もう一度確定するまで、手元AIに聞く・進行の状態には使われません）",
                key=f"reopen_checked_{meeting['record_id']}",
            )
            if st.button(
                "下書きに戻す",
                disabled=not reopen_checked,
                key=f"reopen_meeting_{meeting['record_id']}",
            ):
                try:
                    store.reopen_meeting(
                        customer_id,
                        meeting["record_id"],
                        expected_revision=meeting["revision"],
                    )
                except RevisionConflictError:
                    st.error("他で変更されました。再読込してください。")
                except CustomerStoreError:
                    st.error("下書きに戻せませんでした。再読込してください。")
                else:
                    # 前に開いていた下書きの入力（本文・確認のチェック）を残さない
                    for key in ("draft_details_", "draft_checked_", "reopen_checked_"):
                        st.session_state.pop(f"{key}{meeting['record_id']}", None)
                    st.session_state.shokoban_notice = "下書きに戻しました。上の「下書き」で直して、もう一度確定してください。"
                    st.rerun()


def render_manual_note(store: CustomerStore, customer_id: str) -> None:
    st.subheader("メモを足す")
    note_text = st.text_area("メモ", height=160, key="manual_note_text")
    note_checked = st.checkbox(
        "この内容で記録します", key="manual_note_checked"
    )
    if st.button(
        "記録に足す",
        disabled=not note_checked or not note_text.strip(),
        key="add_manual_note",
    ):
        try:
            store.add_information(
                customer_id,
                text=note_text.strip(),
                kind="note",
                observed_at=jst_now().isoformat(),
                origin="manual",
            )
        except CustomerStoreError:
            st.error("記録に足せませんでした。")
        else:
            st.session_state.shokoban_notice = "記録に足しました。"
            st.rerun()


def render_case_question(
    store: CustomerStore, customer_id: str, paths: MacAppPaths
) -> None:
    st.subheader("手元AIに聞く")
    question = st.text_input("質問", key="case_question")
    if st.button("聞く", key="ask_case"):
        try:
            runtime_config = load_runtime_config(paths.runtime_config)
            llm_settings = prepare_local_llm(
                runtime_config, st.session_state.get("mac_llm_alias")
            )
        except LLMRuntimeError as exc:
            st.error(str(exc))
        else:
            try:
                answer = answer_case_question(
                    store,
                    customer_id,
                    question,
                    generate=lambda prompt: generate_local_text(
                        llm_settings, prompt, num_predict=600, timeout=300
                    ),
                )
            except CaseChatError as exc:
                st.error(str(exc))
            except LocalLLMError:
                st.error("手元AIが応答しませんでした。")
            else:
                st.session_state.shokoban_answer = {
                    "customer_id": customer_id,
                    "question": question.strip(),
                    "answer": answer.answer,
                    "used": answer.used,
                }

    stored = st.session_state.get("shokoban_answer")
    if stored and stored["customer_id"] == customer_id:
        # 手元AIの答えと使った記録は外から来た文字。Markdown として解釈させない。
        st.text(f"答え: {stored['answer']}")
        if stored["used"]:
            st.text("使った記録: " + " / ".join(stored["used"]))
        answer_checked = st.checkbox(
            "この答えをメモに足す", key="answer_note_checked"
        )
        if st.button(
            "メモに足す",
            disabled=not answer_checked,
            key="add_answer_note",
        ):
            text = f"質問: {stored['question']}\n答え: {stored['answer']}"
            observed_at = jst_now().isoformat()
            try:
                try:
                    store.add_information(
                        customer_id,
                        text=text,
                        kind="note",
                        observed_at=observed_at,
                        origin="ai_assisted",
                    )
                except RecordValidationError:
                    # ai_assisted は提案の受理を通らないと使えないので、手で足す扱いにする。
                    store.add_information(
                        customer_id,
                        text=text,
                        kind="note",
                        observed_at=observed_at,
                        origin="manual",
                    )
            except CustomerStoreError:
                st.error("記録に足せませんでした。")
            else:
                st.session_state.shokoban_answer = None
                st.session_state.shokoban_notice = "答えをメモに足しました。"
                st.rerun()


def _meeting_body(meeting: dict) -> str:
    """打ち合わせ1件の本文（要約・詳細・決まったこと・次にやること）."""
    parts = [str(meeting.get("summary") or ""), str(meeting.get("details") or "")]
    decisions = [str(value) for value in meeting.get("decisions") or []]
    next_actions = [str(value) for value in meeting.get("next_actions") or []]
    if decisions:
        parts.append("決まったこと: " + " / ".join(decisions))
    if next_actions:
        parts.append("次にやること: " + " / ".join(next_actions))
    return "\n".join(part for part in parts if part.strip())


def status_records(
    store: CustomerStore, customer_id: str, confirmed: list[dict]
) -> list[tuple[str, str]]:
    """確定済みの打ち合わせ（新しい順）と現在の情報を (見出し, 本文) にする."""
    dated: list[tuple[str, str, str]] = []
    for meeting in confirmed:
        occurred_at = str(meeting.get("occurred_at") or "")
        heading = f"{occurred_at[:10]} {meeting.get('title') or ''}".strip()
        dated.append((occurred_at, heading, _meeting_body(meeting)))
    for metadata, body in store.current_information(customer_id):
        observed_at = str(metadata.get("observed_at") or "")
        dated.append((observed_at, f"メモ {observed_at[:10]}".strip(), body))
    dated.sort(key=lambda item: item[0], reverse=True)
    return [(heading, body) for _stamp, heading, body in dated]


def default_status_date(confirmed: list[dict]) -> date:
    """期限の日付入力の初期値（最新の確定済み打ち合わせの日、無ければ今日）."""
    if confirmed:
        try:
            return date.fromisoformat(str(confirmed[0]["occurred_at"])[:10])
        except ValueError:
            pass
    return jst_now().date()


def set_status_widgets(
    customer_id: str, status: ProgressStatus, fallback_date: date
) -> None:
    """表示用の selectbox・date_input の値を、新しい結果に合わせる."""
    st.session_state[f"status_stage_{customer_id}"] = status.stage
    st.session_state[f"status_next_task_{customer_id}"] = status.next_task
    st.session_state[f"status_mover_{customer_id}"] = status.mover
    due_kind_key = f"status_due_kind_{customer_id}"
    if status.due in ("なし", "不明"):
        st.session_state[due_kind_key] = status.due
        st.session_state.pop(f"status_due_date_{customer_id}", None)
    else:
        st.session_state[due_kind_key] = "日付"
        st.session_state[f"status_due_date_{customer_id}"] = date.fromisoformat(
            status.due
        )


def build_status_with_local_llm(
    paths: MacAppPaths,
    mode: str,
    customer_id: str,
    label: str | None,
    records: list[tuple[str, str]],
    meeting_date: date,
) -> None:
    """手元AIで進行の状態を作り、保存して（呼び名があれば）そのまま共有する."""
    try:
        runtime_config = load_runtime_config(paths.runtime_config)
        llm_settings = prepare_local_llm(
            runtime_config, st.session_state.get("mac_llm_alias")
        )
    except LLMRuntimeError as exc:
        st.error(str(exc))
        return

    def generate(prompt: str) -> str:
        return generate_local_text(
            llm_settings, prompt, num_predict=1500, timeout=300
        )

    try:
        result = infer_progress_status(
            records, meeting_date=meeting_date, generate=generate
        )
    except LocalLLMError:
        st.error("手元AIが応答しませんでした。")
        return
    except (ProgressStatusError, ValueError):
        st.error("進行の状態を作れませんでした。")
        return

    now = jst_now()
    try:
        save_status_record(
            status_record_path(paths, mode, customer_id),
            result.status,
            result.evidence,
            raw_ok=result.raw_ok,
            updated=now,
            source="手元AI",
            inferred=result.status,
        )
    except (MacAppError, OSError, ProgressStatusError):
        st.error("進行の状態を保存できませんでした。")
        return

    set_status_widgets(customer_id, result.status, meeting_date)

    if label is None:
        st.warning("呼び名が決まっていないので共有していません。")
        return
    try:
        write_share_file(
            share_dir_for_mode(paths, mode),
            label,
            result.status,
            updated=now,
            source="手元AI",
            case_id=customer_id,
        )
    except ProgressStatusError as exc:
        st.error(escape_label(str(exc)))
    except OSError:
        st.error("共有ファイルを書けませんでした。")
    else:
        st.success("進行の状態を共有しました（ほかのエージェントが読む場所）")


def render_progress_status(
    store: CustomerStore, customer_id: str, paths: MacAppPaths, mode: str
) -> None:
    """進行の状態（選択肢と日付だけ）を作り、共有し、直す画面."""
    st.subheader("進行の状態")
    labels_path = status_local_dir_for_mode(paths, mode) / "share-labels.json"
    label = load_share_labels(labels_path).get(customer_id)

    if label is None:
        st.caption(
            "ほかのエージェントに見えるのはこの呼び名だけです。"
            "お客様の正式名は使わないでください。"
        )
        new_label = st.text_input(
            "共有用の呼び名", key=f"share_label_input_{customer_id}"
        )
        if st.button("呼び名を決める", key=f"save_share_label_{customer_id}"):
            try:
                normalized = validate_share_label(new_label)
            except ProgressStatusError as exc:
                st.error(str(exc))
            else:
                try:
                    save_share_label(labels_path, customer_id, normalized)
                except MacAppError as exc:
                    st.error(str(exc))
                except OSError:
                    st.error("呼び名を保存できませんでした。")
                else:
                    label = normalized
                    st.success("共有用の呼び名を決めました。")
                    st.text(f"共有用の呼び名: {normalized}")
    else:
        st.text(f"共有用の呼び名: {label}")

    confirmed = [
        meeting
        for meeting in store.list_meetings(customer_id)
        if meeting.get("meeting_status") == "confirmed"
    ]
    records = status_records(store, customer_id, confirmed)
    fallback_date = default_status_date(confirmed)

    if st.button("進行の状態を作る", key=f"build_progress_status_{customer_id}"):
        if not confirmed:
            st.info("確定した打ち合わせがまだありません。")
        else:
            try:
                meeting_date = date.fromisoformat(
                    str(confirmed[0]["occurred_at"])[:10]
                )
            except ValueError:
                st.error("打ち合わせの日付を読めませんでした。")
            else:
                build_status_with_local_llm(
                    paths, mode, customer_id, label, records, meeting_date
                )

    saved = load_status_record(status_record_path(paths, mode, customer_id))
    if saved is None:
        current = ProgressStatus("不明", "不明", "不明", "不明")
        evidence = {field: "" for field in FIELDS}
    else:
        current = saved["status"]
        evidence = saved["evidence"]

    st.markdown("**今の状態**")

    stage_key = f"status_stage_{customer_id}"
    if stage_key not in st.session_state:
        st.session_state[stage_key] = current.stage
    stage = st.selectbox("今どこまで", STAGES, key=stage_key)

    task_key = f"status_next_task_{customer_id}"
    if task_key not in st.session_state:
        st.session_state[task_key] = current.next_task
    next_task = st.selectbox("次にやること", TASKS, key=task_key)

    due_kind_key = f"status_due_kind_{customer_id}"
    if due_kind_key not in st.session_state:
        st.session_state[due_kind_key] = (
            current.due if current.due in ("なし", "不明") else "日付"
        )
    due_kind = st.selectbox("期限", ("日付", "なし", "不明"), key=due_kind_key)
    if due_kind == "日付":
        due_date_key = f"status_due_date_{customer_id}"
        if due_date_key not in st.session_state:
            if current.due in ("なし", "不明"):
                st.session_state[due_date_key] = fallback_date
            else:
                st.session_state[due_date_key] = date.fromisoformat(current.due)
        chosen_due = st.date_input("期限の日付", key=due_date_key)
        due = chosen_due.isoformat() if isinstance(chosen_due, date) else "不明"
    else:
        due = due_kind

    mover_key = f"status_mover_{customer_id}"
    if mover_key not in st.session_state:
        st.session_state[mover_key] = current.mover
    mover = st.selectbox("次に動く側", MOVER, key=mover_key)

    for field in FIELDS:
        if evidence.get(field):
            # 根拠は手元AIの引用。Markdown として解釈させず、そのまま出す。
            st.text(f"根拠（{field}、手元にだけ）: {evidence[field]}")

    if st.button("この内容で共有する", key=f"share_edited_status_{customer_id}"):
        if label is None:
            st.warning("呼び名が決まっていないので共有していません。")
            return
        edited = ProgressStatus(
            stage=stage, next_task=next_task, due=due, mover=mover
        )
        now = jst_now()
        try:
            write_share_file(
                share_dir_for_mode(paths, mode),
                label,
                edited,
                updated=now,
                source="人が修正",
                case_id=customer_id,
            )
        except ProgressStatusError as exc:
            st.error(escape_label(str(exc)))
            return
        except OSError:
            st.error("共有ファイルを書けませんでした。")
            return
        try:
            save_status_record(
                status_record_path(paths, mode, customer_id),
                edited,
                evidence,
                raw_ok=saved["raw_ok"] if saved is not None else False,
                updated=now,
                source="人が修正",
                inferred=saved["inferred"] if saved is not None else None,
            )
        except (MacAppError, OSError, ProgressStatusError):
            st.error("進行の状態を保存できませんでした。")
        else:
            st.success("進行の状態を共有しました（ほかのエージェントが読む場所）")


def render_customer(store: CustomerStore, paths: MacAppPaths, mode: str) -> None:
    customer_id = st.session_state.get("shokoban_customer_id")
    if not customer_id:
        st.header("案件ファイル")
        st.info("案件一覧からお客様を選んでください。")
        return
    try:
        customer = store.get_customer(customer_id)
    except CustomerStoreError:
        st.header("案件ファイル")
        st.error("選んだお客様が見つかりません。案件一覧から選び直してください。")
        st.session_state.shokoban_customer_id = None
        return

    notice = st.session_state.pop("shokoban_notice", None)
    if notice:
        st.success(notice)

    st.header("案件ファイル")
    st.subheader(escape_label(customer["display_name"]))
    render_drafts(store, customer_id)
    render_records(store, customer_id)
    render_manual_note(store, customer_id)
    render_case_question(store, customer_id, paths)
    render_progress_status(store, customer_id, paths, mode)


def _name_choice_key(customer_id: str, term: str) -> str:
    return f"name_choice_{customer_id}_{term}"


def _name_common_key(customer_id: str, term: str) -> str:
    return f"name_common_{customer_id}_{term}"


def _clear_name_choices() -> None:
    """名前の選択肢の入力状態を消す（候補を出し直すとき）."""
    for key in [k for k in st.session_state if str(k).startswith(("name_choice_", "name_common_"))]:
        del st.session_state[key]


def _save_name_decisions(
    paths: MacAppPaths,
    mode: str,
    state: dict,
    choices: dict[str, str],
    *,
    sure_only: bool,
    common: set[str] | None = None,
) -> None:
    """決めた分（あとで以外）を辞書に保存し、保存した行を一覧から外す（全体共通にした語は共通の辞書へ）."""
    try:
        decisions = collect_decisions(state["result"], choices, sure_only=sure_only)
        if not decisions:
            st.info("保存する語がありません（「あとで」の語は保存しません）。")
            return
        dict_path = name_dict_path(paths, mode, state["customer_id"])
        applied, updated = save_decisions(
            dict_path,
            state["result"],
            decisions,
            common_path=common_dict_path(paths, mode),
            common_terms={term for term in (common or set()) if term in decisions},
        )
    except NameReviewError as exc:
        st.error(escape_label(str(exc)))
        return
    for term in decisions:
        st.session_state.pop(_name_choice_key(state["customer_id"], term), None)
        st.session_state.pop(_name_common_key(state["customer_id"], term), None)
    state["result"] = updated
    state["notice"] = (
        f"辞書に保存しました（伏せる語{applied.masked}・伏せない語{applied.passed}）。"
    )
    st.rerun()


def render_names(store: CustomerStore, paths: MacAppPaths, mode: str) -> None:
    """名前の確認（本番データは ``SHOKOBAN_REVIEW_REAL=1`` のときだけ開く）: 候補を出し、1語ずつ決めて辞書に保存する."""
    st.header("名前の確認")
    try:
        require_review_mode(mode)
    except NameReviewError as exc:
        st.info(str(exc))
        return
    st.info(
        "人名・社名の候補を、網・手元AI・GiNZA で探します。"
        "ここで決めていない語（「あとで」のまま・押さなかった語）は、送るときにすべて伏せたままです。"
    )

    drafts = list_draft_meetings(store, include_confirmed=review_real_enabled(mode))
    if not drafts:
        st.info("確認できる下書きの打ち合わせがありません。")
        return
    index = st.selectbox(
        "打ち合わせ",
        list(range(len(drafts))),
        format_func=lambda i: escape_label(
            f"{drafts[i]['display_name']} ・ {str(drafts[i]['occurred_at'])[:10]} ・ {drafts[i]['title']}"
            + ("（確定済み）" if drafts[i].get("meeting_status") == "confirmed" else "")
        ),
        key="name_review_meeting",
    )
    draft = drafts[index]

    if st.button("候補を出す", key="name_review_run"):
        _clear_name_choices()
        st.session_state.name_review = None
        try:
            customer = store.get_customer(draft["customer_id"])
            with st.spinner("候補を探しています（手元AIと GiNZA を使います）"):
                result = build_review(
                    paths,
                    mode,
                    draft["customer_id"],
                    meeting_sentences(draft),
                    alias=st.session_state.get("mac_llm_alias"),
                    company_name=customer["display_name"],
                    aliases=list(customer.get("aliases") or []),
                )
        except (NameReviewError, CustomerStoreError) as exc:
            st.error(escape_label(str(exc)))
        else:
            st.session_state.name_review = {
                "record_id": draft["record_id"],
                "customer_id": draft["customer_id"],
                "result": result,
                "notice": None,
            }

    state = st.session_state.get("name_review")
    if not state or state["record_id"] != draft["record_id"]:
        return
    notice = state.pop("notice", None)
    if notice:
        st.success(notice)
    result = state["result"]

    if result.ginza_error:
        st.warning(
            "GiNZA の候補は出せませんでした（" + escape_label(result.ginza_error) + "）。"
            "ほかの出どころの候補だけです。見つかった候補は伏せたままです。"
        )
    if result.llm_error:
        st.warning(
            "手元AIの候補は出せませんでした（" + escape_label(result.llm_error) + "）。"
            "網と GiNZA の候補だけです。"
        )
    if result.llm_failed_chunks:
        st.warning(f"手元AIが答えられなかった区切りが{result.llm_failed_chunks}個あります。")

    if not result.rows:
        st.info("確認する候補はありません。")
        return

    choices: dict[str, str] = {}
    common: set[str] = set()
    labels = list(CHOICE_LABELS)
    for judgment in JUDGMENT_ORDER:
        rows = [row for row in result.rows if row.judgment == judgment]
        if not rows:
            continue
        st.subheader(f"{judgment}（{len(rows)}）")
        for row in rows:
            columns = st.columns([2, 4, 5])
            columns[0].text(f"{row.term}（{row.kind}）")
            key = _name_choice_key(state["customer_id"], row.term)
            choices[row.term] = columns[1].radio(
                "種類",
                labels,
                index=labels.index(row.default_choice),
                horizontal=True,
                key=key,
                label_visibility="collapsed",
            )
            if columns[1].checkbox(
                "全体共通",
                key=_name_common_key(state["customer_id"], row.term),
                help="ほかのお客様でも同じ扱いにします（共通の辞書に保存）。",
            ):
                common.add(row.term)
            for example in row.examples:
                columns[2].html(highlight_html(example, row.term))

    total, decided, later = count_choices(choices)
    st.text(f"候補{total}・確定{decided}・あとで{later}")
    st.caption(
        "「あとで」の語と、保存しなかった語は、送るときに伏せたままです。"
        "「全体共通」にした語は、ほかのお客様でも同じ扱いになります"
        "（共通で「名前じゃない」にした語でも、確かの候補として出たときは、ここでもう一度確かめます）。"
    )
    sure_button, save_button = st.columns(2)
    if sure_button.button("確かの分をまとめて確定", key="name_review_sure"):
        _save_name_decisions(paths, mode, state, choices, sure_only=True, common=common)
    if save_button.button("保存", type="primary", key="name_review_save"):
        _save_name_decisions(paths, mode, state, choices, sure_only=False, common=common)


def _send_hide_key(record_id: str, sent: int) -> str:
    return f"send_hide_{record_id}_{sent}"


def _send_ok_key(record_id: str, sent: int) -> str:
    return f"send_ok_{record_id}_{sent}"


def _region_tagger(paths: MacAppPaths):
    """区間の札付け役（手元の12B）。使えないときは偽物（区間はすべて伏せる）と、その理由."""
    try:
        cfg = load_runtime_config(paths.runtime_config)
        name = tagger_model()
        if name not in (getattr(cfg, "models", {}) or {}):
            raise MaskR3Error(
                "札付けに使う手元AIが設定にありません。"
                "SHOKOBAN_TAGGER_MODEL か runtime.json の models に、その名前を入れてください。"
            )
        settings = prepare_local_llm(cfg, name)
        return make_llama_tagger(settings.model), name, None
    except (LLMRuntimeError, MaskR3Error, LocalLLMError) as exc:
        return dry_tagger, "dry", str(exc) or "手元AIを使えませんでした。"


def _sentence_html(sent: int, text: str, flagged: bool, reasons: tuple[str, ...]) -> str:
    body = html.escape(text)
    note = ""
    if flagged:
        note = (
            '<div style="font-size:0.8rem;opacity:0.8;margin-top:0.15rem">'
            + html.escape("・".join(reasons))
            + "</div>"
        )
    style = (
        "padding:0.35rem 0.5rem;border-radius:0.3rem;"
        + ("background:rgba(255,196,0,0.22);border-left:4px solid #e0a800;" if flagged else "")
    )
    return f'<div style="{style}"><b>{sent}</b>　{body}{note}</div>'


def render_send(store: CustomerStore, paths: MacAppPaths, mode: str) -> None:
    """送る前の確認（本番データは ``SHOKOBAN_REVIEW_REAL=1`` のときだけ開く）: 伏せたあとの全文を見て、確認済みにする."""
    st.header("送る前の確認")
    try:
        require_review_mode(mode)
    except NameReviewError as exc:
        st.info(str(exc))
        return
    st.info(
        "クラウドに送る前に、伏せたあとの文をすべて見ます。漏れていそうな文は黄色で目立たせます"
        "（全体の1割まで）。気になる文は「さらに伏せる」を選んでから、確認済みにしてください。"
    )

    drafts = list_draft_meetings(store, include_confirmed=review_real_enabled(mode))
    if not drafts:
        st.info("確認できる下書きの打ち合わせがありません。")
        return
    index = st.selectbox(
        "打ち合わせ",
        list(range(len(drafts))),
        format_func=lambda i: escape_label(
            f"{drafts[i]['display_name']} ・ {str(drafts[i]['occurred_at'])[:10]} ・ {drafts[i]['title']}"
            + ("（確定済み）" if drafts[i].get("meeting_status") == "confirmed" else "")
        ),
        key="send_review_meeting",
    )
    draft = drafts[index]

    names = st.session_state.get("name_review")
    if not names or names["record_id"] != draft["record_id"]:
        st.warning("先に「名前の確認」で、この打ち合わせの候補を出してください（候補は伏せる語に使います）。")
        return

    if st.button("伏せて確かめる", key="send_review_run"):
        st.session_state.send_review = None
        try:
            customer = store.get_customer(draft["customer_id"])
            name_dict, _ = load_review_dicts(
                name_dict_path(paths, mode, draft["customer_id"]), common_dict_path(paths, mode)
            )
            with st.spinner("伏せています（手元AIで区間の札を付けます）"):
                tagger, tagger_model, note = _region_tagger(paths)
                review = prepare_send_review(
                    meeting_sentences(draft),
                    name_dict=name_dict,
                    candidates=names["result"].candidates,
                    company_name=customer["display_name"],
                    aliases=list(customer.get("aliases") or []),
                    tagger=tagger,
                    tagger_model=tagger_model,
                    tagger_note=note,
                    ok_phrases=load_ok_phrases(ok_phrases_path(paths, mode)),
                )
        except (SendReviewError, NameReviewError, NameDictError, CustomerStoreError, OSError) as exc:
            st.error(escape_label(str(exc)))
        else:
            for key in [k for k in st.session_state if str(k).startswith(("send_hide_", "send_ok_", "send_word_", "send_review_read_all", "cloud_memo_edit_"))]:
                del st.session_state[key]
            st.session_state.send_review = {"record_id": draft["record_id"], "review": review}

    state = st.session_state.get("send_review")
    if not state or state["record_id"] != draft["record_id"]:
        return
    review = state["review"]
    if review.tagger_note:
        st.warning("区間の札付けを使えなかったので、区間はすべて伏せました（" + escape_label(review.tagger_note) + "）。")

    st.text(f"全{review.total}文・目立たせ{review.flagged}文・短い相づち{review.short}文（畳んであります）")
    only_flagged = st.checkbox("目立たせた文だけ表示", key="send_review_only_flagged")
    hide: list[int] = []
    ok_rows: list = []
    words: list[HideWord] = []
    kind_labels = {"名前": "人名", "社名・店名・団体": "社名・店名・団体"}

    def show(container, row) -> None:
        columns = container.columns([9, 2])
        columns[0].html(_sentence_html(row.sent, row.text, row.flagged, row.reasons))
        if columns[1].checkbox("さらに伏せる", key=_send_hide_key(draft["record_id"], row.sent)):
            hide.append(row.sent)
            rid = draft["record_id"]
            word = columns[0].text_input(
                "辞書に入れる語（入れるなら。次から自動で伏せます）", key=f"send_word_{rid}_{row.sent}"
            )
            if word.strip():
                kind = columns[0].radio(
                    "種類", list(kind_labels), horizontal=True, key=f"send_word_kind_{rid}_{row.sent}"
                )
                common = columns[0].checkbox("全体共通", key=f"send_word_common_{rid}_{row.sent}")
                words.append(HideWord(row.sent, word.strip(), kind_labels[kind], common))
        if row.ok_phrases and columns[1].checkbox(
            "問題なし",
            key=_send_ok_key(draft["record_id"], row.sent),
            help="この言い回しを、次から目立たせません（全体共通。伏せ方は変わりません）。",
        ):
            ok_rows.append(row)

    short_rows = []
    for row in review.rows:
        key = _send_hide_key(draft["record_id"], row.sent)
        if only_flagged and not row.flagged:
            if st.session_state.get(key):
                hide.append(row.sent)
            if st.session_state.get(_send_ok_key(draft["record_id"], row.sent)) and row.ok_phrases:
                ok_rows.append(row)
            continue
        if row.short:
            short_rows.append(row)
            continue
        show(st, row)
    if short_rows:
        with st.expander(f"短い相づち {len(short_rows)}文（開いて見られます）"):
            for row in short_rows:
                show(st, row)

    try:
        path = approval_path(paths, mode, draft["customer_id"], draft["record_id"])
        final_texts(review, hide)  # 文番号の検査（ない番号なら止める）
        final = final_masked(review, hide)  # クラウドに送る中身全体
        approved = check_payload_approval(path, final)
        saved = load_approval(path)
    except SendReviewError as exc:
        st.error(escape_label(str(exc)))
        return
    if approved:
        st.success("この内容で確認済みです（" + escape_label(str(saved["reviewed_at"])[:16]) + "）。")
    elif saved is not None:
        st.warning("前に確認したときと中身が変わっています。もう一度確認してください。")
    st.caption(f"さらに伏せる文: {len(hide)}")
    if ok_rows:
        st.caption(f"「問題なし」にする文: {len(ok_rows)}（確認済みにするときに覚えます）")
    if only_flagged:
        # 目立たせた文だけの表示では、付いたまま灰色に見えないよう外す（全文表示に戻したら付け直す）
        st.session_state["send_review_read_all"] = False
    # 送る中身が変わったら（さらに伏せるを選んだ・外した）、全文を読んだ確認を外して付け直してもらう
    digest_now = payload_digest(final)
    if st.session_state.get("send_review_read_all_for") != digest_now:
        st.session_state["send_review_read_all"] = False
        st.session_state["send_review_read_all_for"] = digest_now
    read_all_box = st.checkbox(
        "伏せたあとの全文を読んだ（目立たせた文だけでなく、畳んだ相づちも含めて）",
        key="send_review_read_all",
        disabled=only_flagged,
        help="付けないと確認済みにできません。「目立たせた文だけ表示」を切って、全文を見てから付けてください。",
    )
    # 目立たせた文だけ表示している間は、付けていないものとして扱う（全文を読んだことにならない）
    read_all = bool(read_all_box) and not only_flagged
    if only_flagged:
        st.caption("「目立たせた文だけ表示」を切ると、全文を読んだ確認を付けられます。")
    if st.button("確認済みにする", type="primary", key="send_review_approve"):
        try:
            if not read_all:
                raise SendReviewError("伏せたあとの全文を読んだ確認を付けてください（目立たせた文だけの表示は切ってください）。")
            checked_words = check_hide_words(review, words)
            save_approval(path, review, hide, reviewed_at=jst_now(), read_all=True)
            if checked_words:
                save_hide_words(
                    name_dict_path(paths, mode, draft["customer_id"]), common_dict_path(paths, mode), checked_words
                )
            phrases = [phrase for row in ok_rows if row.sent not in hide for phrase in row.ok_phrases]
            if phrases:
                add_ok_phrases(ok_phrases_path(paths, mode), phrases)
        except (SendReviewError, NameReviewError, MacAppError, OSError) as exc:
            st.error(escape_label(str(exc)))
        else:
            st.rerun()

    _render_cloud_memo(paths, mode, draft, final, path, approved)


def _meeting_date(draft: dict) -> date | None:
    try:
        return date.fromisoformat(str(draft.get("occurred_at", ""))[:10])
    except ValueError:
        return None


def _show_memo_sections(text: str) -> None:
    """5欄（と道筋）を、欄ごとの見出し（件数つき）で出す。本文は ``st.text`` だけで、Markdown にしない.

    道筋は一番上に番号つき（「こちら」が話したのは見出しで分かるので、行頭の ``[こちら]`` は省く）。
    まだ決まっていないことは件数が多いので畳む。見出しの無い文字（前に保存した版や、人が自由に直した版）も
    そのまま出す。
    """
    preface, sections = split_memo_sections(text)
    if preface:
        st.text("\n".join(preface))
    for name, entries in sections:
        title = f"■ {name}（{len(entries)}）"
        if name == F_PATH:
            own = speaker_mark("支援側")
            lines = [
                f"{number}. {entry[len(own):] if entry.startswith(own) else entry}"
                for number, entry in enumerate(entries, start=1)
            ]
        else:
            lines = [f"・{entry}" for entry in entries]
        body = "\n".join(lines) if lines else "なし"
        if name == F_UNDECIDED:
            with st.expander(escape_label(title)):
                st.text(body)
        else:
            st.text(title)
            st.text(body)


def _render_cloud_memo(paths: MacAppPaths, mode: str, draft: dict, final, approval_file: Path, approved: bool) -> None:
    """5欄を作る（読む役）: 確認済みの中身だけを送り、人が直して保存する."""
    try:
        reader, reader_name = cloud_reader_from_env()
        mpath = memo_path(paths, mode, draft["customer_id"], draft["record_id"])
        memo = load_memo(mpath)
    except (CloudMemoError, SendReviewError) as exc:
        st.subheader("5欄を作る（読む役）")
        st.error(escape_label(str(exc)))
        return
    is_local = reader is not None and reader_is_local(reader_name)
    st.subheader("5欄を作る（手元の読む役）" if is_local else "5欄を作る（クラウドの読む役）")
    edit_key = f"cloud_memo_edit_{draft['record_id']}"
    if reader is None:
        st.info("クラウドの読む役は止めてあります（環境変数 SHOKOBAN_CLOUD_READER で有効にします）。")
    elif not approved:
        st.caption("先に「確認済みにする」を押してください（全文を読んだ確認が要ります）。確認した中身だけを送ります。")
    else:
        if is_local:
            st.caption(f"手元の読む役: {escape_label(reader_name)}。確認済みの中身を、この PC の中のサーバーにだけ送ります。")
        else:
            st.caption(f"読む役: {escape_label(reader_name)}。確認済みの中身だけを送ります。")
        counts_usage = reader_name != READER_FAKE  # 偽の読む役は外へ出ないので数えない
        month = jst_now().strftime("%Y-%m")
        if counts_usage:
            try:
                used_now = load_usage(usage_path(paths, mode)).get(month, 0)
            except (CloudMemoError, MacAppError) as exc:
                st.error(escape_label(str(exc)))
                return
            st.caption(f"今月の読む役: {used_now}/{monthly_limit()}回")
        same = memo is not None and memo.get("payload_sha256") == payload_digest(final)
        edited_before = memo is not None and memo.get("edited", memo.get("shared")) != memo.get("shared")
        may_make = True
        if same or edited_before:
            # 作り直すと前の5欄（人が直した版を含む）を上書きするので、はっきり選んだときだけ作る
            label = "同じ中身でもう一度作る（読む役をもう一度呼びます" if same else "前の5欄を捨てて作る（"
            if edited_before:
                label += "。人が直した版は消えます" if same else "人が直した版は消えます"
            label += "）"
            # 作った5欄ごとに別の鍵にする（作り直したあと、選んだままにならない）
            may_make = st.checkbox(label, key=f"cloud_memo_redo_{draft['record_id']}_{memo.get('created_at', '')}")
        if st.button("この内容で5欄を作る", key="cloud_memo_make", disabled=not may_make):
            try:
                if counts_usage:
                    reserve_reader_call(usage_path(paths, mode), month, monthly_limit())
                with st.spinner("読む役に送って、5欄を作っています"):
                    made = make_memo(final, approval_file=approval_file, reader=reader, meeting_date=_meeting_date(draft))
                    # 返事が来たら、画面に触れる前（spinner を閉じる前）に保存する。待つ間に画面が
                    # 読み直されると、次の st の呼び出しでこの回が打ち切られ、答えが捨てられるため
                    save_memo(mpath, made)
            except (CloudMemoError, SendReviewError, MacAppError, OSError) as exc:
                st.error(escape_label(str(exc)))
            else:
                st.session_state.pop(edit_key, None)
                st.rerun()
    if memo is None or not approved or memo.get("payload_sha256") != payload_digest(final):
        return
    st.caption("読む役: " + escape_label(str(memo.get("reader_model") or "不明")))
    saved_edit = memo.get("edited")
    is_edited = type(saved_edit) is str and saved_edit != memo["shared"]
    if is_edited:
        st.caption("人が直した版を表示しています（読む役が作った元の版ではありません）。")
    _show_memo_sections(saved_edit if is_edited else memo["shared"])
    with st.expander("全体を直す"):
        edited = st.text_area(
            "全体（直して保存できます）",
            value=str(saved_edit if type(saved_edit) is str else memo["shared"]),
            height=360,
            key=edit_key,
        )
        if st.button("直して保存", key="cloud_memo_save_edited"):
            try:
                save_edited(mpath, edited)
            except (CloudMemoError, OSError) as exc:
                st.error(escape_label(str(exc)))
            else:
                # 作り直しの案内（人が直した版は消えます）を今の保存に合わせるため、読み直す
                st.session_state.cloud_memo_saved = draft["record_id"]
                st.rerun()
    if st.session_state.get("cloud_memo_saved") == draft["record_id"]:
        del st.session_state["cloud_memo_saved"]
        st.success("直した版を保存しました。")
    _render_share_memo(paths, mode, draft, mpath)


def _render_share_memo(paths: MacAppPaths, mode: str, draft: dict, mpath: Path) -> None:
    """確認済みの5欄（直した版があればそちら）を、ほかのエージェントが読める share/進行メモ/ に渡す."""
    st.subheader("共有フォルダに渡す")
    labels_path = status_local_dir_for_mode(paths, mode) / "share-labels.json"
    label = load_share_labels(labels_path).get(draft["customer_id"])
    if label is None:
        st.info("先に「案件ファイル」の「進行の状態」で、共有用の呼び名を決めてください（ほかのエージェントに見えるのは呼び名だけです）。")
        return
    try:
        share_dir = share_dir_for_mode(paths, mode)
        target = share_memo_path(share_dir, label, draft["record_id"], _meeting_date(draft))
    except (CloudMemoError, MacAppError) as exc:
        st.error(escape_label(str(exc)))
        return
    shown = target.relative_to(share_dir)
    st.caption("渡す先: share/" + escape_label(str(shown)) + "（伏せたあとの5欄だけ。元の記録は渡しません）")
    if target.is_file():
        st.caption("渡し済みです。直したあとにもう一度押すと、直した版で上書きします。")
    if st.button("この5欄を共有フォルダに渡す", key="cloud_memo_share"):
        try:
            memo = load_memo(mpath)
            if memo is None:
                raise CloudMemoError("渡す5欄がありません。")
            share_memo(
                share_dir, label, memo,
                record_id=draft["record_id"], meeting_date=_meeting_date(draft), shared_at=jst_now(),
            )
        except (CloudMemoError, MacAppError, OSError) as exc:
            st.error(escape_label(str(exc)))
        else:
            st.success("渡しました。ほかのエージェントには share/" + escape_label(str(shown)) + " を読むよう伝えてください。")


def render_llm(paths: MacAppPaths) -> None:
    st.header("手元AI")
    try:
        runtime_config = load_runtime_config(paths.runtime_config)
    except LLMRuntimeError as exc:
        st.error(str(exc))
        return

    status = runtime_status(runtime_config)
    if status["running"]:
        st.success("手元AIは動いています。")
        st.text(f"別名: {status['alias'] or '不明'}")
    else:
        st.info("手元AIは止まっています。")
    if status["idle_seconds"] is not None:
        st.caption(f"最後に使ってから {int(status['idle_seconds'] // 60)} 分")

    if st.button("手元AIを今すぐ止める", key="stop_local_llm"):
        try:
            llm_runtime.stop(runtime_config)
        except LLMRuntimeError as exc:
            st.error(str(exc))
        else:
            st.success("手元AIを止めました。")

    aliases = list(runtime_config.models)
    current_alias = st.session_state.get("mac_llm_alias")
    if current_alias not in aliases:
        current_alias = runtime_config.default_model
    chosen_alias = st.selectbox(
        "手元AIのモデル",
        aliases,
        index=aliases.index(current_alias),
        key="mac_llm_alias_select",
    )
    st.session_state.mac_llm_alias = chosen_alias

    idle = getattr(runtime_config, "idle_minutes", None)
    if type(idle) in (int, float) and idle > 0:
        idle_text = f"{idle:g}分"
    else:
        idle_text = "しばらく"
    st.info(f"手元AIは使う時だけ自動で起動し、{idle_text}使わなければ止まります。")


def main() -> None:
    mode = current_mode()
    paths = mac_paths_or_stop()

    st.markdown(f"データモード：{MODE_LABELS[mode]}")

    if "mac_llm_alias" not in st.session_state:
        try:
            runtime_config = load_runtime_config(paths.runtime_config)
        except LLMRuntimeError:
            pass
        else:
            st.session_state.mac_llm_alias = runtime_config.default_model

    store = open_store(paths, mode)
    try:
        ensure_inbox(paths)
    except MacAppError as exc:
        st.error("受け取りフォルダを用意できませんでした: " + escape_label(str(exc)))
        st.stop()

    if "shokoban_screen" not in st.session_state:
        st.session_state.shokoban_screen = "cases"
    render_nav()

    screen = st.session_state.shokoban_screen
    if screen == "customer":
        render_customer(store, paths, mode)
    elif screen == "names":
        render_names(store, paths, mode)
    elif screen == "send":
        render_send(store, paths, mode)
    elif screen == "llm":
        render_llm(paths)
    else:
        render_cases(store, paths, mode)


main()
