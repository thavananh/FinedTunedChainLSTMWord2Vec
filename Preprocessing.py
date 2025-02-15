from calendar import c
from concurrent.futures import ThreadPoolExecutor, as_completed
import string
import unicodedata
import regex as re
import numpy as np
import pandas as pd
from underthesea import sent_tokenize, word_tokenize, text_normalize
from loky import get_reusable_executor


class VietnameseTextPreprocessor:
    def __init__(self, stopwords_path):
        self.uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
        self.unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"
        self.dicchar = self.loaddicchar()
        self.bang_nguyen_am = [
            ["a", "à", "á", "ả", "ã", "ạ"],
            ["ă", "ằ", "ắ", "ẳ", "ẵ", "ặ"],
            ["â", "ầ", "ấ", "ẩ", "ẫ", "ậ"],
            ["e", "è", "é", "ẻ", "ẽ", "ẹ"],
            ["ê", "ề", "ế", "ể", "ễ", "ệ"],
            ["i", "ì", "í", "ỉ", "ĩ", "ị"],
            ["o", "ò", "ó", "ỏ", "õ", "ọ"],
            ["ô", "ồ", "ố", "ổ", "ỗ", "ộ"],
            ["ơ", "ờ", "ớ", "ở", "ỡ", "ợ"],
            ["u", "ù", "ú", "ủ", "ũ", "ụ"],
            ["ư", "ừ", "ứ", "ử", "ữ", "ự"],
            ["y", "ỳ", "ý", "ỷ", "ỹ", "ỵ"],
        ]
        self.nguyen_am_to_ids = {}
        for idx, vowel_group in enumerate(self.bang_nguyen_am):
            for jdx, vowel in enumerate(vowel_group):
                self.nguyen_am_to_ids[vowel] = (idx, jdx)
        self.stopwords_path = stopwords_path
        
        self.list_rare_words_1 = [
            "fraction",
            "altera",
            "quad",
            "cnpm",
            "kaydotvn",
            "daadotuitdotedudotvn",
            "bohm",
            "dotnet",
            "visual studio code",
            "visual studio",
            "visual code",
            "visual",
            "mutton quad",
            "dreamweaver",
            "vertical fragmentation",
            "i-ta-li-a",
            "đbcl",
            "crt",
            "javabeans",
            "if",
            "for",
            "while",
            "struct",
            "pl",
            "windows phone",
            "pm",
            "assembly",
            "scrum_project",
            "scrum project",
            "css",
            "jquery",
            "vector",
            "gb",
            "ram",
            "ram",
            "...",
            "codefun",
            "egov",
            "java",
            "vmware",
            "placement",
            "fork",
            "round robin",
            "patin",
            "pattern",
            "serverside",
            "dotnet",
            "c++",
            "javascript",
            "ch",
            "uit khoa",
            "proteus",
            "console",
            "form",
            "vật lý học",
            "aep",
            "servlet",
            "skype",
            "doubledot",
            "poster",
            "everything",
            "amp",
            "app",
            "seo",
            "app",
            "progressive web",
            "xim",
            "win server",
            "đối với",
            "p",
            "s",
            "p s",
            "dropbox",
            "bạn",
            "cplusplus",
            "socket",
            "sub",
            "switch",
            "tôi",
            "em",
            "các em",
            "tụi em",
            "chúng em",
            "các bạn",
            "tụi em",
            "chúng em" "severside",
            "network programing with csharp",
            "dfd",
            "naives bayes",
            "naive",
            "bayes",
            "cs",
            "js",
            "max",
            "elg",
            "fix",
            "proxy",
            "hub",
            "bridge",
            "switch",
            "windows",
            "turnitindotcom",
            "extensive reading",
            "reading",
            "extensive",
            "search",
            "quick",
            "file header",
            "paper",
            "directx",
            "windows",
            "linux",
            "vote",
            "itdotf",
            "router",
            "silverlight",
            "đa luồng",
            "crack",
            "wrede",
            "dbpedia",
            "ontology",
            "tmf",
            "vhdl",
            "hdl",
            "jsp",
            "pđt",
            "lisp",
            "json",
            "cpp",
            "itp",
            "forum",
            "embeded",
            "system",
            "embeded system",
            "embedded",
            "titanium",
            "blackbery",
            "zun",
            "phonegap",
            "tizen",
            "je",
            "mediafire",
            "toeic",
            "ghz",
            "cpu",
            "module",
            "datapath",
            "papers",
            "daa",
            "dijktra",
            "oracal",
            "database",
            "access",
            "netbean",
            "facebook",
            "hackerrankdotcom",
            "sort",
            "multiagent",
            "th",
            "contemn",
            "dbms",
            "html",
            "php",
            "heapsort",
            "khmtdotuitdotedudotvn",
            "vdotv",
            "engine",
            "download",
            "input",
            "output",
            "wtf",
            "forum",
            "poison",
            "uit",
            "career",
            "như",
            "version",
            "outdoor",
            "coursedotuitdotedudotvn",
            "mini",
            "matlab",
            "standford",
            "name",
            "size",
            "framework",
            "ucla",
            "comment",
            "is",
            "we",
            "serverside",
            "cassette",
            "ios",
            "android",
            "scrum",
            "itdote",
            "xml",
            "photo",
            "down",
            "unikey",
            "3dsmax",
            "firmware",
            "km",
            "hackerrank",
            "projectbase",
            "er",
            "gay",
            "feed",
            "mác – lênin",
            "mác",
            "lênin",
            "coursedotuitdotedudotvn",
            "nfc",
            "chip",
            "full",
            "oi",
            "ht",
            "ubuntu",
            "linux",
            "it",
            "wecode",
            "code",
            "oop",
            "hướng đối tượng",
            "cho",
            "để",
            "macbook",
            "a_z",
            "moodle",
            "trang",
            "stack",
            "queue",
            "bảng băm",
            "chính trị",
            "progressive",
            "at",
            "boss",
            "ic",
            "pdf",
            "driver",
            "room",
            "link",
            "thpt",
            "thcs",
            "murray",
            "store",
            "severside",
            "hes",
            "dsmax",
            "header",
            "prolog",
            "pro",
            "ida",
            "youtube",
            "hit",
            "google",
            "facebook",
            "twitter",
            "instagram",
            "linkedin",
            "pinterest",
            "tumblr",
            "reddit",
            "snapchat",
            "whatsapp",
            "viber",
            "line",
            "zalo",
            "wechat",
            "telegram",
            "skype",
            "vimeo",
            "flickr",
            "periscope",
            "twitch",
            "soundcloud",
            "spotify",
            "shazam",
            "apple",
            "samsung",
            "nokia",
            "sony",
            "lg",
            "htc",
            "huawei",
            "oppo",
            "vivo",
            "xiaomi",
            "meizu",
            "asus",
            "acer",
            "dell",
            "hp",
            "lenovo",
            "microsoft",
            "google",
            "amazon",
            "ebay",
            "alibaba",
            "paypal",
            "visa",
            "mastercard",
            "american express",
            "jcb",
            "smartphone",
            "word",
            "lap",
            "dev",
            "mfc",
            "protues",
            "unit",
            "srum",
            "dev",
            "are",
            "mfc",
            "protues",
            "ielts",
            "bit",
            "oracle",
            "người dùng",
            "user",
            "bus",
            "de",
            "...",
            "virus",
            "sql",
            "closeness",
            "topdown",
            "computer",
            "hhhhhhhhhhhhhhhhhhhhhhh",
            "betweenness",'các em', 'chúng em', 'tụi em', 'các bạn', 'tụi em', 'sinh_viên', 'là', 'và', 'thì', 'vì', 'mà', 'của', 'khi', 'như', 'lại', 'đó', 'đây', 'kia', 'ấy', 'sẽ', 'mình', 'nếu', 'vậy', 'rồi', 'với', 'bởi', 'mà', 'ấy', 'kia', 'sẽ', 'đó', 'dù', 'tuy','sinh viên', 'học sinh', 'giáo viên', 'giảng viên', 'cc', 'tma'
        ]
        self.list_rare_words = [ # chi so tf_idf
            "fraction",
            "altera",
            "quad",
            "cnpm",
            "kaydotvn",
            "daadotuitdotedudotvn",
            "bohm",
            "dotnet",
            "visual studio code",
            "visual studio",
            "visual code",
            "visual",
            "mutton quad",
            "dreamweaver",
            "vertical fragmentation",
            "i-ta-li-a",
            "đbcl",
            "crt",
            "javabeans",
            "if",
            "for",
            "while",
            "struct",
            "pl",
            "windows phone",
            "pm",
            "assembly",
            "scrum_project",
            "scrum project",
            "css",
            "jquery",
            "vector",
            "gb",
            "ram",
            "ram",
            "...",
            "codefun",
            "egov",
            "java",
            "vmware",
            "placement",
            "fork",
            "round robin",
            "patin",
            "pattern",
            "serverside",
            "dotnet",
            "c++",
            "javascript",
            "ch",
            "uit khoa",
            "proteus",
            "console",
            "form",
            "vật lý học",
            "aep",
            "servlet",
            "skype",
            "doubledot",
            "poster",
            "everything",
            "amp",
            "app",
            "seo",
            "app",
            "progressive web",
            "xim",
            "win server",
            "đối với",
            "p",
            "s",
            "p s",
            "dropbox",
            "bạn",
            "cplusplus",
            "socket",
            "sub",
            "switch",
            "tôi",
            "em",
            "các em",
            "tụi em",
            "chúng em",
            "các bạn",
            "tụi em",
            "chúng em" "severside",
            "network programing with csharp",
            "dfd",
            "naives bayes",
            "naive",
            "bayes",
            "cs",
            "js",
            "max",
            "elg",
            "fix",
            "proxy",
            "hub",
            "bridge",
            "switch",
            "windows",
            "turnitindotcom",
            "extensive reading",
            "reading",
            "extensive",
            "search",
            "quick",
            "file header",
            "paper",
            "directx",
            "windows",
            "linux",
            "vote",
            "itdotf",
            "router",
            "silverlight",
            "đa luồng",
            "crack",
            "wrede",
            "dbpedia",
            "ontology",
            "tmf",
            "vhdl",
            "hdl",
            "jsp",
            "pđt",
            "lisp",
            "json",
            "cpp",
            "itp",
            "forum",
            "embeded",
            "system",
            "embeded system",
            "embedded",
            "titanium",
            "blackbery",
            "zun",
            "phonegap",
            "tizen",
            "je",
            "mediafire",
            "toeic",
            "ghz",
            "cpu",
            "module",
            "datapath",
            "papers",
            "daa",
            "dijktra",
            "oracal",
            "database",
            "access",
            "netbean",
            "facebook",
            "hackerrankdotcom",
            "sort",
            "multiagent",
            "th",
            "contemn",
            "dbms",
            "html",
            "php",
            "heapsort",
            "khmtdotuitdotedudotvn",
            "vdotv",
            "engine",
            "download",
            "input",
            "output",
            "wtf",
            "forum",
            "poison",
            "uit",
            "career",
            "như",
            "version",
            "outdoor",
            "coursedotuitdotedudotvn",
            "mini",
            "matlab",
            "standford",
            "name",
            "size",
            "framework",
            "ucla",
            "comment",
            "is",
            "we",
            "serverside",
            "cassette",
            "ios",
            "android",
            "scrum",
            "itdote",
            "xml",
            "photo",
            "down",
            "unikey",
            "3dsmax",
            "firmware",
            "km",
            "hackerrank",
            "projectbase",
            "er",
            "gay",
            "feed",
            "mác – lênin",
            "mác",
            "lênin",
            "coursedotuitdotedudotvn",
            "nfc",
            "chip",
            "full",
            "oi",
            "ht",
            "ubuntu",
            "linux",
            "it",
            "wecode",
            "code",
            "oop",
            "hướng đối tượng",
            "cho",
            "để",
            "macbook",
            "a_z",
            "moodle",
            "trang",
            "stack",
            "queue",
            "bảng băm",
            "chính trị",
            "progressive",
            "at",
            "boss",
            "ic",
            "pdf",
            "driver",
            "room",
            "link",
            "thpt",
            "thcs",
            "murray",
            "store",
            "severside",
            "hes",
            "dsmax",
            "header",
            "prolog",
            "pro",
            "ida",
            "youtube",
            "hit",
            "google",
            "facebook",
            "twitter",
            "instagram",
            "linkedin",
            "pinterest",
            "tumblr",
            "reddit",
            "snapchat",
            "whatsapp",
            "viber",
            "line",
            "zalo",
            "wechat",
            "telegram",
            "skype",
            "vimeo",
            "flickr",
            "periscope",
            "twitch",
            "soundcloud",
            "spotify",
            "shazam",
            "apple",
            "samsung",
            "nokia",
            "sony",
            "lg",
            "htc",
            "huawei",
            "oppo",
            "vivo",
            "xiaomi",
            "meizu",
            "asus",
            "acer",
            "dell",
            "hp",
            "lenovo",
            "microsoft",
            "google",
            "amazon",
            "ebay",
            "alibaba",
            "paypal",
            "visa",
            "mastercard",
            "american express",
            "jcb",
            "smartphone",
            "word",
            "lap",
            "dev",
            "mfc",
            "protues",
            "unit",
            "srum",
            "dev",
            "are",
            "mfc",
            "protues",
            "ielts",
            "bit",
            "oracle",
            "người dùng",
            "user",
            "bus",
            "de",
            "...",
            "virus",
            "sql",
            "closeness",
            "topdown",
            "computer",
            "hhhhhhhhhhhhhhhhhhhhhhh",
            "betweenness", 'các em', 'chúng em', 'tụi em', 'các bạn', 'tụi em', 'sinh_viên', 'là', 'và', 'thì', 'vì', 'mà', 'của', 'khi', 'như', 'lại', 'đó', 'đây', 'kia', 'ấy', 'sẽ', 'mình', 'nếu', 'vậy', 'rồi', 'với', 'bởi', 'mà', 'ấy', 'kia', 'sẽ', 'đó', 'dù', 'tuy','sinh viên', 'học sinh', 'giáo viên', 'giảng viên', 'cc', 'tma', 
        ]
        self.mapping_dict = {
            "thầy giáo": "giảng viên",
            "cô giáo": "giảng viên",
            "thầy": "giảng viên",
            "cô": "giảng viên",
            "giáo viên": "giảng viên",
            "vói": "với",
            "giời": "giờ",
            "nhiệt hình": "nhiệt tình",
            "side": "slide",
            "tân tình": "tận tình",
            "teacher": "giảng viên",
            "sadcolon": "colonsad",
            "vi dụ": "ví dụ",
            "easy": "dễ",
            "so vời": "so với",
            "tâp": "tập",
            "av": "tiếng anh",
            "nhannh": "nhanh",
            "h": "giờ",
            "đc": "được",
            "dc": "được",
            "smilesmile": "smile",
            "tron": "trong",
            "thướng": "hướng",
            "nghung": "nhưng",
            "chon": "chọn",
            "them": "thêm",
            "day": "dạy",
            "midterm": "giữa kỳ",
            "vi": "vì",
            "quýêt": "quyết",
            "teamwork": "làm việc nhóm",
            "over time": "quá giờ",
            "overtime": "quá giờ",
            "ot": "quá giờ",
            "outcome": "mục tiêu",
            "basic": "cơ bản",
            "check": "kiểm tra",
            "gmail": "email",
            "mail": "email",
            "teen": "trẻ",
            "style": "phong cách",
            "topic": "chủ đề",
            "file": "tài liệu",
            "slides": "tài liệu",
            "slide": "tài liệu",
            "giáo trình": "tài liệu",
            "slile": "tài liệu",
            "silde": "tài liệu",
            "web": "website",
            "online": "website",
            "nope": "không",
            "class": "lớp học",
            "fitted": "phù hợp",
            "chair": "ghế",
            "table": "bàn",
            "grammar": "ngữ pháp",
            "speaking": "kỹ năng nói",
            "listening": "kỹ năng nghe",
            "listenning": "kỹ năng nghe",
            "reading": "kỹ năng đọc",
            "writing": "kỹ năng viết",
            "good": "tốt",
            "english": "tiếng anh",
            "bad": "tệ",
            "feedback": "phản hồi",
            "elab": "lab",
            "group": "nhóm",
            "mic": "microphone",
            "micro": "microphone",
            "debate": "thảo luận",
            "test": "kiểm tra",
            "thưc": "thực",
            "courses": "khóa học",
            "course": "khóa học",
            "funny": "vui vẻ",
            "internet": "mạng",
            "perfect": "hoàn hảo",
            "quoa": "qua",
            "thanks": "cảm ơn",
            "thankss": "cảm ơn",
            "ok": "ổn",
            "sư phạm": "giảng dạy",
            "bigsmile": "smile",
            "smallsmile": "smile",
            "doublesurprise": "surprise",
            "show": "cho xem",
            "requirement": "yêu cầu",
            "assignments": "bài tập",
            "famous": "nổi tiếng",
            # 'store': 'cửa hàng',
            "presentation": "trình chiếu",
            "book": "sách",
            "ebook": "sách",
            "bạnem": "bạn",
            "not": "không",
            "cute": "dễ thương",
            "project": "đồ án",
            "dự án": "đồ án",
            "quỳên": "quyền",
            "colonsad": "sad",
            "miss": "thiếu",
            "hoat": "hoạt",
            "upload": "post",
            "design": "thiết kế",
            "home": "nhà",
            "give": "cho",
            "given": "cho",
            "sưc": "sức",
            "traning": "giảng dạy",
            "reply": "trả lời",
            "eemail": "email",
            "submit": "nộp",
            "hihi": "smile",
            "team": "nhóm",
            "very": "rất",
            "like": "thích",
            "chat": "trò chuyện",
            "lap": "laptop",
            "free": "miễn phí",
            "wed": "website",
            "update": "cập nhật",
            "pc": "máy tính",
            "ko": "không",
            "stepbystep": "từng bước",
            "again": "lần nữa",
            "outcomes": "mục tiêu",
            "outcomes": "mục tiêu",
            "lovelove": "love",
            "textbook": "tài liệu",
            "speak": "nói",
            "tasks": "bài tập",
            "sociable": "thân thiện",
            "shes": "giảng viên",
            "level": "cấp độ",
            'e-mail':'email',
            'step-by-step':'từng bước',
            'aproach': 'tiếp cận',
            't_t': 'sad',
            'trc': 'trước',
        }
        self.stopwords_small = self.load_stopwords()

    def loaddicchar(self):
        dic = {}
        char1252 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
            "|"
        )
        charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
            "|"
        )
        for i in range(len(char1252)):
            dic[char1252[i]] = charutf8[i]
        return dic

    def covert_unicode(self, txt):
        return re.sub(
            r"à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ",
            lambda x: self.dicchar[x.group()],
            txt,
        )

    def is_valid_vietnam_word(self, word):
        chars = list(word)
        nguyen_am_index = -1
        for index, char in enumerate(chars):
            x, y = self.nguyen_am_to_ids.get(char, (-1, -1))
            if x != -1:
                if nguyen_am_index == -1:
                    nguyen_am_index = index
                else:
                    if index - nguyen_am_index != 1:
                        return False
                    nguyen_am_index = index
        return True

    def chuan_hoa_dau_tieng_viet(self, word):
        if not self.is_valid_vietnam_word(word):
            return word
        chars = list(word)
        dau_cau = 0
        nguyen_am_index = []
        qu_or_gi = False
        for index, char in enumerate(chars):
            x, y = self.nguyen_am_to_ids.get(char, (-1, -1))
            if x == -1:
                continue
            elif x == 9:  # check qu
                if index != 0 and chars[index - 1] == "q":
                    chars[index] = "u"
                    qu_or_gi = True
            elif x == 5:  # check gi
                if index != 0 and chars[index - 1] == "g":
                    chars[index] = "i"
                    qu_or_gi = True
            if y != 0:
                dau_cau = y
                chars[index] = self.bang_nguyen_am[x][0]
                if not qu_or_gi or index != 1:
                    nguyen_am_index.append(index)
        if len(nguyen_am_index) < 2:
            if qu_or_gi:
                if len(chars) == 2:
                    x, y = self.nguyen_am_to_ids.get(chars[1], (-1, -1))
                    if x != -1:
                        chars[1] = self.bang_nguyen_am[x][dau_cau]
                else:
                    x, y = self.nguyen_am_to_ids.get(chars[2], (-1, -1))
                    if x != -1:
                        chars[2] = self.bang_nguyen_am[x][dau_cau]
                    else:
                        chars[1] = (
                            self.bang_nguyen_am[5][dau_cau]
                            if chars[1] == "i"
                            else self.bang_nguyen_am[9][dau_cau]
                        )
                return "".join(chars)
            return word
        for index in nguyen_am_index:
            x, y = self.nguyen_am_to_ids.get(chars[index], (-1, -1))
            if x == 4 or x == 8:  # Example indices for specific vowels like 'ê', 'ơ'
                chars[index] = self.bang_nguyen_am[x][dau_cau]
        if len(nguyen_am_index) == 2:
            if nguyen_am_index[-1] == len(chars) - 1:
                x, y = self.nguyen_am_to_ids.get(chars[nguyen_am_index[0]], (-1, -1))
                if x != -1:
                    chars[nguyen_am_index[0]] = self.bang_nguyen_am[x][dau_cau]
            else:
                x, y = self.nguyen_am_to_ids.get(chars[nguyen_am_index[1]], (-1, -1))
                if x != -1:
                    chars[nguyen_am_index[1]] = self.bang_nguyen_am[x][dau_cau]
        else:
            x, y = self.nguyen_am_to_ids.get(chars[nguyen_am_index[1]], (-1, -1))
            if x != -1:
                chars[nguyen_am_index[1]] = self.bang_nguyen_am[x][dau_cau]
        return "".join(chars)

    def chuan_hoa_dau_cau_tieng_viet(self, sentence):
        sentence = sentence.lower()
        words = sentence.split()
        for index, word in enumerate(words):
            cw = re.sub(r"(^\p{P}*)([\p{L}]+)(\p{P}*$)", r"\1/\2/\3", word).split("/")
            if len(cw) == 3:
                cw[1] = self.chuan_hoa_dau_tieng_viet(cw[1])
                words[index] = "".join(cw)
        return " ".join(words)

    def load_stopwords(self):
        with open(self.stopwords_path, "r", encoding="utf-8") as file:
            stopwords_small = [line.strip() for line in file.readlines()]
        # stopwords_small = [line.strip().replace(" ", "_") for line in stopwords_small]

        # stopwords_small = [line.strip().replace(" ", "_") for line in stopwords_small]
        stopwords_small.append("dot")
        # stopwords_small.append('giảng viên')
        for item in self.list_rare_words_1:
            stopwords_small.append(item)

        print("Stopwords: ", len(stopwords_small))
        print(stopwords_small)
        return stopwords_small

    def preprocess_text_vietnamese_to_tokens(self, text, isReturnTokens=True, isUsingDash=True, isSimple=False):
        text = unicodedata.normalize('NFKC', str(text)).lower()
        text = text_normalize(text) 
        text = text.translate(str.maketrans('', '', string.punctuation)) # xoa dau cau
        
        # Thay thế từ ngữ theo mapping_dict
        for original, replacement in self.mapping_dict.items():
            text = re.sub(r'\b' + re.escape(original) + r'\b', replacement, text)
        
        # Xóa emoji
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002500-\U00002BEF"  # Chinese characters
            "\U00002702-\U000027B0"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "\U0001f926-\U0001f937"
            "\U00010000-\U0010ffff"
            "\u2640-\u2642"
            "\u2600-\u2B55"
            "\u200d"
            "\u23cf"
            "\u23e9"
            "\u231a"
            "\ufe0f"  # dingbats
            "\u3030"
            "]+", flags=re.UNICODE
        )
        text = emoji_pattern.sub('', text)

        # Thay thế 'colon' và 'doubledot' nếu có
        text = re.sub(r'\bcolon(\w+)\b', r'\1', text)
        text = re.sub(r'\bdoubledot(\w+)\b', r'\1', text)
        
        # Loại bỏ các tiền tố không cần thiết
        for prefix in ['wzjwz', 'wwzjwz', 'dot']:
            text = re.sub(r'\b' + re.escape(prefix) + r'\w*\b', '', text)

        # Loại bỏ các từ hiếm
        rarewords_pattern = r'\b(?:' + '|'.join(re.escape(word) for word in self.list_rare_words) + r')\b'
        text = re.sub(rarewords_pattern, '', text)
        
        # Tokenize và loại bỏ số
        text = " ".join([word_tokenize(sent, format='text') for sent in sent_tokenize(text)]) if isUsingDash else " ".join([word for sent in sent_tokenize(text) for word in word_tokenize(sent)])
        text = re.sub(r'\d+', '', text)
        text = re.sub('\s+', ' ', text)
        stopwords_set = set(self.stopwords_small)
        if not isSimple:
            # Tokenize lại
            tokens = word_tokenize(text, format='text').split() if isUsingDash else word_tokenize(text)
            
            # Loại bỏ từ dừng (stopwords)
            
            tokens = [token for token in tokens if token not in stopwords_set and token.strip()]
            tokens = [self.mapping_dict.get(token, token) for token in tokens]
            # Trả về kết quả
        else:
            
            tokens = text.split()
            tokens = [token for token in tokens if token not in stopwords_set and token.strip()]
            tokens = [self.mapping_dict.get(token, token) for token in tokens]
        
        if isReturnTokens:
            return tokens
        return ' '.join(tokens)


    def preprocess_texts_concurrently(
        self, texts, isReturnTokens=True, isUsingDash=True
    ):
        """
        Preprocess multiple texts concurrently using loky's ProcessPoolExecutor.
        """
        preprocessed_texts = []
        with get_reusable_executor(max_workers=24) as executor:
            future_to_text = {
                executor.submit(
                    self.preprocess_text_vietnamese_to_tokens,
                    text,
                    isReturnTokens,
                    isUsingDash,
                ): text
                for text in texts
            }
            for future in as_completed(future_to_text):
                try:
                    preprocessed_text = future.result()
                    preprocessed_texts.append(preprocessed_text)
                except Exception as e:
                    print(f"An error occurred while preprocessing text: {e}")
        return preprocessed_texts
