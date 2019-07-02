TAGGED_REFS = """
SELECT rf.ref_name
FROM repositories r
    NATURAL JOIN refs rf
    NATURAL JOIN commits c
WHERE r.repository_id = '%s' AND is_tag(rf.ref_name)
ORDER BY c.committer_when;
"""

FILE_INFO = """
SELECT rf.ref_name,
    cf.file_path,
    cf.blob_hash,
    LANGUAGE(cf.file_path) as lang
FROM repositories r
    NATURAL JOIN refs rf
    NATURAL JOIN commit_files cf
WHERE r.repository_id = '%s'
    AND rf.ref_name in (%s)
    AND lang in (%s);
"""

FILE_CONTENT = """
SELECT DISTINCT
    cf.file_path,
    cf.blob_hash,
    LANGUAGE(cf.file_path) as lang,
    f.blob_content
FROM repositories r
    NATURAL JOIN refs rf
    NATURAL JOIN commit_files cf
    NATURAL JOIN files f
WHERE r.repository_id = '%s'
    AND is_tag(rf.ref_name)
    AND rf.ref_name in (%s)
    AND lang in (%s);
"""
