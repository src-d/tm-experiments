TAGGED_VERSIONS = """
SELECT rf.ref_name
FROM repositories r
    NATURAL JOIN refs rf
WHERE r.repository_id = '%s' AND is_tag(rf.ref_name);
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
    AND is_tag(rf.ref_name)
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
    AND lang in (%s)
"""

IDENTIFIERS = "identifiers"
LITERALS = "literals"
COMMENTS = "comments"

IDENTIFIER_XPATH = "//uast:Identifier"
LITERAL_XPATH = "//uast:String"
COMMENT_XPATH = "//uast:Comment"

IDENTIFIER_KEY = "Name"
LITERAL_KEY = "Value"
COMMENT_KEY = "Text"

FEATURE_MAPPING = {
    IDENTIFIERS: {"xpath": IDENTIFIER_XPATH, "key": IDENTIFIER_KEY},
    LITERALS: {"xpath": LITERAL_XPATH, "key": LITERAL_KEY},
    COMMENTS: {"xpath": COMMENT_XPATH, "key": COMMENT_KEY},
}

SUPPORTED_LANGUAGES = [
    "C#",
    "C++",
    "C",
    "Cuda",
    "OpenCL",
    "Metal",
    "Bash",
    "Shell",
    "Go",
    "Java",
    "JavaScript",
    "JS",
    "JSX",
    "PHP",
    "Python",
    "Ruby",
    "TypeScript",
]
