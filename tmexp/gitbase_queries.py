from typing import List


def get_tagged_refs_sql(*, repository_id: str) -> str:
    return (
        """
        SELECT rf.ref_name
        FROM repositories r
            NATURAL JOIN refs rf
            NATURAL JOIN commits c
        WHERE r.repository_id = '%s' AND is_tag(rf.ref_name)
        ORDER BY c.committer_when;
        """
        % repository_id
    )


def get_file_info_sql(
    *, repository_id: str, ref_names: List[str], langs: List[str], exclude_vendors: bool
) -> str:
    return """
        SELECT rf.ref_name,
            cf.file_path,
            cf.blob_hash,
            LANGUAGE(cf.file_path) AS lang
        FROM repositories r
            NATURAL JOIN refs rf
            NATURAL JOIN commit_files cf
        WHERE r.repository_id = '%s'
            AND rf.ref_name IN (%s)
            AND lang IN (%s)
            %s;
        """ % (
        repository_id,
        ",".join("'%s'" % ref_name for ref_name in ref_names),
        ",".join("'%s'" % lang for lang in langs),
        "AND NOT IS_VENDOR(cf.file_path)" if exclude_vendors else "",
    )


def get_file_content_sql(
    *, repository_id: str, ref_names: List[str], langs: List[str], exclude_vendors: bool
) -> str:
    return """
        SELECT DISTINCT
            cf.file_path,
            cf.blob_hash,
            LANGUAGE(cf.file_path) AS lang,
            f.blob_content
        FROM repositories r
            NATURAL JOIN refs rf
            NATURAL JOIN commit_files cf
            NATURAL JOIN files f
        WHERE r.repository_id = '%s'
            AND rf.ref_name IN (%s)
            AND lang IN (%s)
            %s;
        """ % (
        repository_id,
        ",".join("'%s'" % ref_name for ref_name in ref_names),
        ",".join("'%s'" % lang for lang in langs),
        "AND NOT IS_VENDOR(cf.file_path)" if exclude_vendors else "",
    )
