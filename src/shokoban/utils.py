"""監査ログ（``audit``）に書くパスの匿名化."""


def sanitize_audit_path(path: str | None) -> str | None:
    """監査ログ用に機微を含むパスを匿名化する.

    customer-vault/sanitized-briefs/memory/exports のいずれかの配下パスは、
    領域名のみ残しファイル名を <redacted> にする.
    """
    if path is None:
        return None
    p = str(path).replace("\\", "/")
    redacted_roots = (
        "customer-vault/",
        "sanitized-briefs/",
        "memory/",
        "exports/",
    )
    for root in redacted_roots:
        if root in p:
            return f"{root[:-1]}/<redacted>"
    exact_roots = ("customer-vault", "sanitized-briefs", "memory", "exports")
    for root in exact_roots:
        if p == root:
            return f"{root}/<redacted>"
    return p
