import numpy as np
import pandas as pd
import re

#from spellchecker import SpellChecker
import time
from multiprocessing import  Pool
#from nltk.stem import WordNetLemmatizer
import random as r
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from toxicClassifier import ToxicClassifier

# https://stackoverflow.com/a/34682849
def untokenize(words):
    """Untokenizing a text undoes the tokenizing operation, restoring
    punctuation and spaces to the places that people expect them to be.
    Ideally, `untokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    """
    text = ' '.join(words)
    step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .', '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
        "can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    return step6.strip()


# https://stackoverflow.com/a/47091490
def decontracted(phrase):
    """Convert contractions like "can't" into "can not"
    """
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    #phrase = re.sub(r"n't", " not", phrase) # resulted in "ca not" when sentence started with "can't"
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


# https://github.com/rishabhverma17/sms_slang_translator/blob/master/slang.txt
slang_abbrev_dict = {
    'AFAIK': 'As Far As I Know',
    'AFK': 'Away From Keyboard',
    'ASAP': 'As Soon As Possible',
    'ATK': 'At The Keyboard',
    'ATM': 'At The Moment',
    'A3': 'Anytime, Anywhere, Anyplace',
    'BAK': 'Back At Keyboard',
    'BBL': 'Be Back Later',
    'BBS': 'Be Back Soon',
    'BFN': 'Bye For Now',
    'B4N': 'Bye For Now',
    'BRB': 'Be Right Back',
    'BRT': 'Be Right There',
    'BTW': 'By The Way',
    'B4': 'Before',
    'B4N': 'Bye For Now',
    'CU': 'See You',
    'CUL8R': 'See You Later',
    'CYA': 'See You',
    'FAQ': 'Frequently Asked Questions',
    'FC': 'Fingers Crossed',
    'FWIW': 'For What It\'s Worth',
    'FU': 'Fuck you',
    'FYI': 'For Your Information',
    'GAL': 'Get A Life',
    'GG': 'Good Game',
    'GN': 'Good Night',
    'GMTA': 'Great Minds Think Alike',
    'GR8': 'Great!',
    'G9': 'Genius',
    'IC': 'I See',
    'ICQ': 'I Seek you',
    'ILU': 'I Love You',
    'IMHO': 'In My Humble Opinion',
    'IMO': 'In My Opinion',
    'IOW': 'In Other Words',
    'IRL': 'In Real Life',
    'KISS': 'Keep It Simple, Stupid',
    'LDR': 'Long Distance Relationship',
    'LMAO': 'Laugh My Ass Off',
    'LOL': 'Laughing Out Loud',
    'LTNS': 'Long Time No See',
    'L8R': 'Later',
    'MTE': 'My Thoughts Exactly',
    'M8': 'Mate',
    'NRN': 'No Reply Necessary',
    'OIC': 'Oh I See',
    'OMG': 'Oh My God',
    'PITA': 'Pain In The Ass',
    'PRT': 'Party',
    'PRW': 'Parents Are Watching',
    'QPSA?': 'Que Pasa?',
    'ROFL': 'Rolling On The Floor Laughing',
    'ROFLOL': 'Rolling On The Floor Laughing Out Loud',
    'ROTFLMAO': 'Rolling On The Floor Laughing My Ass Off',
    'SK8': 'Skate',
    'STATS': 'Your sex and age',
    'STFU': 'Shut the fuck up',
    'ASL': 'Age, Sex, Location',
    'THX': 'Thank You',
    'TTFN': 'Ta-Ta For Now!',
    'TTYL': 'Talk To You Later',
    'U': 'You',
    'U2': 'You Too',
    'U4E': 'Yours For Ever',
    'WB': 'Welcome Back',
    'WTF': 'What The Fuck',
    'WTG': 'Way To Go!',
    'WUF': 'Where Are You From?',
    'W8': 'Wait',
    '7K': 'Sick:-D Laugher'
}


def unslang(text):
    """Converts text like "OMG" into "Oh my God"
    """
    if text.upper() in slang_abbrev_dict.keys():
        return slang_abbrev_dict[text.upper()]
    else:
        return text


# https://gist.github.com/sebleier/554280
stopwords = [
    "a", "about", "above", "after", "again", "against", "ain", "all", "am",
    "an", "and", "any", "are", "aren", "aren't", "as", "at", "be", "because",
    "been", "before", "being", "below", "between", "both", "but", "by", "can",
    "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn",
    "doesn't", "doing", "don", "don't", "down", "during", "each", "few", "for",
    "from", "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have",
    "haven", "haven't", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "isn", "isn't",
    "it", "it's", "its", "itself", "just", "ll", "m", "ma", "me", "mightn",
    "mightn't", "more", "most", "mustn", "mustn't", "my", "myself", "needn",
    "needn't", "no", "nor", "not", "now", "o", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own",
    "re", "s", "same", "shan", "shan't", "she", "she's", "should", "should've",
    "shouldn", "shouldn't", "so", "some", "such", "t", "than", "that",
    "that'll", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "ve", "very", "was", "wasn", "wasn't", "we", "were", "weren",
    "weren't", "what", "when", "where", "which", "while", "who", "whom", "why",
    "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
    "could", "he'd", "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm",
    "i've", "let's", "ought", "she'd", "she'll", "that's", "there's", "they'd",
    "they'll", "they're", "they've", "we'd", "we'll", "we're", "we've",
    "what's", "when's", "where's", "who's", "why's", "would", "able", "abst",
    "accordance", "according", "accordingly", "across", "act", "actually",
    "added", "adj", "affected", "affecting", "affects", "afterwards", "ah",
    "almost", "alone", "along", "already", "also", "although", "always",
    "among", "amongst", "announce", "another", "anybody", "anyhow", "anymore",
    "anyone", "anything", "anyway", "anyways", "anywhere", "apparently",
    "approximately", "arent", "arise", "around", "aside", "ask", "asking",
    "auth", "available", "away", "awfully", "b", "back", "became", "become",
    "becomes", "becoming", "beforehand", "begin", "beginning", "beginnings",
    "begins", "behind", "believe", "beside", "besides", "beyond", "biol",
    "brief", "briefly", "c", "ca", "came", "cannot", "can't", "cause", "causes",
    "certain", "certainly", "co", "com", "come", "comes", "contain",
    "containing", "contains", "couldnt", "date", "different", "done",
    "downwards", "due", "e", "ed", "edu", "effect", "eg", "eight", "eighty",
    "either", "else", "elsewhere", "end", "ending", "enough", "especially",
    "et", "etc", "even", "ever", "every", "everybody", "everyone", "everything",
    "everywhere", "ex", "except", "f", "far", "ff", "fifth", "first", "five",
    "fix", "followed", "following", "follows", "former", "formerly", "forth",
    "found", "four", "furthermore", "g", "gave", "get", "gets", "getting",
    "give", "given", "gives", "giving", "go", "goes", "gone", "got", "gotten",
    "h", "happens", "hardly", "hed", "hence", "hereafter", "hereby", "herein",
    "heres", "hereupon", "hes", "hi", "hid", "hither", "home", "howbeit",
    "however", "hundred", "id", "ie", "im", "immediate", "immediately",
    "importance", "important", "inc", "indeed", "index", "information",
    "instead", "invention", "inward", "itd", "it'll", "j", "k", "keep", "keeps",
    "kept", "kg", "km", "know", "known", "knows", "l", "largely", "last",
    "lately", "later", "latter", "latterly", "least", "less", "lest", "let",
    "lets", "like", "liked", "likely", "line", "little", "'ll", "look",
    "looking", "looks", "ltd", "made", "mainly", "make", "makes", "many", "may",
    "maybe", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might",
    "million", "miss", "ml", "moreover", "mostly", "mr", "mrs", "much", "mug",
    "must", "n", "na", "name", "namely", "nay", "nd", "near", "nearly",
    "necessarily", "necessary", "need", "needs", "neither", "never",
    "nevertheless", "new", "next", "nine", "ninety", "nobody", "non", "none",
    "nonetheless", "noone", "normally", "nos", "noted", "nothing", "nowhere",
    "obtain", "obtained", "obviously", "often", "oh", "ok", "okay", "old",
    "omitted", "one", "ones", "onto", "ord", "others", "otherwise", "outside",
    "overall", "owing", "p", "page", "pages", "part", "particular",
    "particularly", "past", "per", "perhaps", "placed", "please", "plus",
    "poorly", "possible", "possibly", "potentially", "pp", "predominantly",
    "present", "previously", "primarily", "probably", "promptly", "proud",
    "provides", "put", "q", "que", "quickly", "quite", "qv", "r", "ran",
    "rather", "rd", "readily", "really", "recent", "recently", "ref", "refs",
    "regarding", "regardless", "regards", "related", "relatively", "research",
    "respectively", "resulted", "resulting", "results", "right", "run", "said",
    "saw", "say", "saying", "says", "sec", "section", "see", "seeing", "seem",
    "seemed", "seeming", "seems", "seen", "self", "selves", "sent", "seven",
    "several", "shall", "shed", "shes", "show", "showed", "shown", "showns",
    "shows", "significant", "significantly", "similar", "similarly", "since",
    "six", "slightly", "somebody", "somehow", "someone", "somethan",
    "something", "sometime", "sometimes", "somewhat", "somewhere", "soon",
    "sorry", "specifically", "specified", "specify", "specifying", "still",
    "stop", "strongly", "sub", "substantially", "successfully", "sufficiently",
    "suggest", "sup", "sure", "take", "taken", "taking", "tell", "tends", "th",
    "thank", "thanks", "thanx", "thats", "that've", "thence", "thereafter",
    "thereby", "thered", "therefore", "therein", "there'll", "thereof",
    "therere", "theres", "thereto", "thereupon", "there've", "theyd", "theyre",
    "think", "thou", "though", "thoughh", "thousand", "throug", "throughout",
    "thru", "thus", "til", "tip", "together", "took", "toward", "towards",
    "tried", "tries", "truly", "try", "trying", "ts", "twice", "two", "u", "un",
    "unfortunately", "unless", "unlike", "unlikely", "unto", "upon", "ups",
    "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using",
    "usually", "v", "value", "various", "'ve", "via", "viz", "vol", "vols",
    "vs", "w", "want", "wants", "wasnt", "way", "wed", "welcome", "went",
    "werent", "whatever", "what'll", "whats", "whence", "whenever",
    "whereafter", "whereas", "whereby", "wherein", "wheres", "whereupon",
    "wherever", "whether", "whim", "whither", "whod", "whoever", "whole",
    "who'll", "whomever", "whos", "whose", "widely", "willing", "wish",
    "within", "without", "wont", "words", "world", "wouldnt", "www", "x", "yes",
    "yet", "youd", "youre", "z", "zero", "a's", "ain't", "allow", "allows",
    "apart", "appear", "appreciate", "appropriate", "associated", "best",
    "better", "c'mon", "c's", "cant", "changes", "clearly", "concerning",
    "consequently", "consider", "considering", "corresponding", "course",
    "currently", "definitely", "described", "despite", "entirely", "exactly",
    "example", "going", "greetings", "hello", "help", "hopefully", "ignored",
    "inasmuch", "indicate", "indicated", "indicates", "inner", "insofar",
    "it'd", "keep", "keeps", "novel", "presumably", "reasonably", "second",
    "secondly", "sensible", "serious", "seriously", "sure", "t's", "third",
    "thorough", "thoroughly", "three", "well", "wonder", "a", "about", "above",
    "above", "across", "after", "afterwards", "again", "against", "all",
    "almost", "alone", "along", "already", "also", "although", "always", "am",
    "among", "amongst", "amoungst", "amount", "an", "and", "another", "any",
    "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as",
    "at", "back", "be", "became", "because", "become", "becomes", "becoming",
    "been", "before", "beforehand", "behind", "being", "below", "beside",
    "besides", "between", "beyond", "bill", "both", "bottom", "but", "by",
    "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry",
    "de", "describe", "detail", "do", "done", "down", "due", "during", "each",
    "eg", "eight", "either", "eleven", "else", "elsewhere", "empty", "enough",
    "etc", "even", "ever", "every", "everyone", "everything", "everywhere",
    "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five",
    "for", "former", "formerly", "forty", "found", "four", "from", "front",
    "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he",
    "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers",
    "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if",
    "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself",
    "keep", "last", "latter", "latterly", "least", "less", "ltd", "made",
    "many", "may", "me", "meanwhile", "might", "mill", "mine", "more",
    "moreover", "most", "mostly", "move", "much", "must", "my", "myself",
    "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no",
    "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of",
    "off", "often", "on", "once", "one", "only", "onto", "or", "other",
    "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own",
    "part", "per", "perhaps", "please", "put", "rather", "re", "same", "see",
    "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should",
    "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow",
    "someone", "something", "sometime", "sometimes", "somewhere", "still",
    "such", "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thickv", "thin",
    "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within",
    "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves",
    "the", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",
    "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B", "C",
    "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R",
    "S", "T", "U", "V", "W", "X", "Y", "Z", "co", "op", "research-articl",
    "pagecount", "cit", "ibid", "les", "le", "au", "que", "est", "pas", "vol",
    "el", "los", "pp", "u201d", "well-b", "http", "volumtype", "par", "0o",
    "0s", "3a", "3b", "3d", "6b", "6o", "a1", "a2", "a3", "a4", "ab", "ac",
    "ad", "ae", "af", "ag", "aj", "al", "an", "ao", "ap", "ar", "av", "aw",
    "ax", "ay", "az", "b1", "b2", "b3", "ba", "bc", "bd", "be", "bi", "bj",
    "bk", "bl", "bn", "bp", "br", "bs", "bt", "bu", "bx", "c1", "c2", "c3",
    "cc", "cd", "ce", "cf", "cg", "ch", "ci", "cj", "cl", "cm", "cn", "cp",
    "cq", "cr", "cs", "ct", "cu", "cv", "cx", "cy", "cz", "d2", "da", "dc",
    "dd", "de", "df", "di", "dj", "dk", "dl", "do", "dp", "dr", "ds", "dt",
    "du", "dx", "dy", "e2", "e3", "ea", "ec", "ed", "ee", "ef", "ei", "ej",
    "el", "em", "en", "eo", "ep", "eq", "er", "es", "et", "eu", "ev", "ex",
    "ey", "f2", "fa", "fc", "ff", "fi", "fj", "fl", "fn", "fo", "fr", "fs",
    "ft", "fu", "fy", "ga", "ge", "gi", "gj", "gl", "go", "gr", "gs", "gy",
    "h2", "h3", "hh", "hi", "hj", "ho", "hr", "hs", "hu", "hy", "i", "i2", "i3",
    "i4", "i6", "i7", "i8", "ia", "ib", "ic", "ie", "ig", "ih", "ii", "ij",
    "il", "in", "io", "ip", "iq", "ir", "iv", "ix", "iy", "iz", "jj", "jr",
    "js", "jt", "ju", "ke", "kg", "kj", "km", "ko", "l2", "la", "lb", "lc",
    "lf", "lj", "ln", "lo", "lr", "ls", "lt", "m2", "ml", "mn", "mo", "ms",
    "mt", "mu", "n2", "nc", "nd", "ne", "ng", "ni", "nj", "nl", "nn", "nr",
    "ns", "nt", "ny", "oa", "ob", "oc", "od", "of", "og", "oi", "oj", "ol",
    "om", "on", "oo", "oq", "or", "os", "ot", "ou", "ow", "ox", "oz", "p1",
    "p2", "p3", "pc", "pd", "pe", "pf", "ph", "pi", "pj", "pk", "pl", "pm",
    "pn", "po", "pq", "pr", "ps", "pt", "pu", "py", "qj", "qu", "r2", "ra",
    "rc", "rd", "rf", "rh", "ri", "rj", "rl", "rm", "rn", "ro", "rq", "rr",
    "rs", "rt", "ru", "rv", "ry", "s2", "sa", "sc", "sd", "se", "sf", "si",
    "sj", "sl", "sm", "sn", "sp", "sq", "sr", "ss", "st", "sy", "sz", "t1",
    "t2", "t3", "tb", "tc", "td", "te", "tf", "th", "ti", "tj", "tl", "tm",
    "tn", "tp", "tq", "tr", "ts", "tt", "tv", "tx", "ue", "ui", "uj", "uk",
    "um", "un", "uo", "ur", "ut", "va", "wa", "vd", "wi", "vj", "vo", "wo",
    "vq", "vt", "vu", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn",
    "xo", "xs", "xt", "xv", "xx", "y2", "yj", "yl", "yr", "ys", "yt", "zi", "zz"
]


# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
def remove_emoji(text):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


# from: https://www.kaggle.com/shahules/basic-eda-cleaning-and-glove
# maybe a bug, it removes question marks?
#spell = SpellChecker()

def correct_spellings(text):
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)

def remove_urls(text):
    text = clean(r"http\S+", text)
    text = clean(r"www\S+", text)
    text = clean(r"pic.twitter.com\S+", text)

    return text

def clean(reg_exp, text):
    text = re.sub(reg_exp, " ", text)

    # replace multiple spaces with one.
    text = re.sub('\s{2,}', ' ', text)

    return text

#lemmatizer = WordNetLemmatizer()

def clean_all(t, correct_spelling=False, remove_stopwords=False, lemmatize=False):
    from nltk.tokenize import word_tokenize
    # first do bulk cleanup on tokens that don't depend on word tokenization

    # remove xml tags
    t = clean(r"<[^>]+>", t)
    t = clean(r"&lt;", t)
    t = clean(r"&gt;", t)

    # remove URLs
    t = remove_urls(t)

    # https://stackoverflow.com/a/35041925
    # replace multiple punctuation with single. Ex: !?!?!? would become ?
    t = clean(r'[\?\.\!]+(?=[\?\.\!])', t)

    t = remove_emoji(t)

    # expand common contractions like "I'm" "he'll"
    t = decontracted(t)

    # now remove/expand bad patterns per word
    words = word_tokenize(t)

    # remove stopwords
    if remove_stopwords is True:
        words = [w for w in words if not w in stopwords]

    clean_words = []

    for w in words:
        # normalize punctuation
        w = re.sub(r'&', 'and', w)

        # expand slang like OMG = Oh my God
        w = unslang(w)

        if lemmatize is True:
            w = lemmatizer.lemmatize(w)

        clean_words.append(w)

    # join the words back into a full string
    t = untokenize(clean_words)

    if correct_spelling is True:
        # this resulted in lots of lost punctuation - omitting for now. Also greatly speeds things up
        t = correct_spellings(t)

    # finally, remove any non ascii and special characters that made it through
    t = clean(r"[^A-Za-z0-9\.\'!\?,\$]", t)

    return t
def get_texts(sect='train'):
    if(sect == 'test'):
        f = open('./data/offens_eval/olid-test-v1.0.tsv')
        tweets = {}
        header = f.readline()
        line = f.readline().strip()
        while line != None and line != '':
            splt = line.split('\t')
            tweets[splt[0]] = {'tweet':clean_all(splt[1].replace('@USER', '').replace('URL', '')),
                                'is_off':splt[2],
                                'is_tar':splt[3],
                                'tar':splt[4]}

            line = f.readline().strip()
        return tweets
    else:
        f = open('./data/offens_eval/olid-training-v1.0.tsv')
        tweets = {}
        header = f.readline()
        line = f.readline().strip()
        while line != None and line != '':
            splt = line.split('\t')
            tweets[splt[0]] = {'tweet':clean_all(splt[1].replace('@USER', '').replace('URL', '')),
                                'is_off':splt[2],
                                'is_tar':splt[3],
                                'tar':splt[4]}

            line = f.readline().strip()
        return tweets

def add_imp_exp(tweets, sect='train'):
    f = open('./data/offens_eval/offenseval_exp_imp_train.tsv')
    header = f.readline()
    line = f.readline().strip()
    while line != '':
        splt = line.split('\t')
        tweets[splt[0]]['off_imp_exp'] = splt[1] if splt[1] != '0' else "NOT"
        line = f.readline().strip()

    f = open('./data/offens_eval/abuseval_offenseval_train.tsv')
    header = f.readline()
    line = f.readline().strip()
    while line != '':
        splt = line.split('\t')
        tweets[splt[0]]['abuse_imp_exp'] = splt[1] if splt[1] != 'NOTABU' else "NOT"
        line = f.readline().strip()

def sample_tweets(tweets, sample_type, max_len=-1, max_n=-1):
    if(sample_type == 'abuse_exp'):
        new_tweets = [tweets[t]['tweet'] for t in tweets if tweets[t]['abuse_imp_exp'] == 'EXP']
    elif(sample_type == 'abuse_imp'):
        new_tweets = [tweets[t]['tweet'] for t in tweets if tweets[t]['abuse_imp_exp'] == 'IMP']
    elif(sample_type == 'off_exp'):
        new_tweets = [tweets[t]['tweet'] for t in tweets if tweets[t]['off_imp_exp'] == 'EXP']
    elif(sample_type == 'off_imp'):
        new_tweets = [tweets[t]['tweet'] for t in tweets if tweets[t]['off_imp_exp'] == 'IMP']
    elif(sample_type == 'bad'):
        new_tweets = [tweets[t]['tweet'] for t in tweets if (tweets[t]['abuse_imp_exp'] in ['EXP', 'IMP'] or tweets[t]['off_imp_exp'] in ['EXP', 'IMP'])]
    elif(sample_type == 'all'):
        new_tweets = [tweets[t]['tweet'] for t in tweets]
    else:
        new_tweets = []
    if(max_len != -1):
        new_tweets = [x for x in t if len(x) <= max_len]
    r.shuffle(new_tweets)
    if(max_n != -1):
        new_tweets = new_tweets[:max_n]
    return new_tweets

import pandas as pd
import re
def load_wiki_toxic(file, max_n=-1, max_len=-1):
    print("Loading - {}".format(file))
    df = pd.read_csv(file)

    x = df['comment_text'].tolist()
    y = df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].sum(axis=1)
    y = y.where(y == 0, 1).tolist()

    toxic_comments = [x.replace('\n', ' ').replace('|','') for i,x in enumerate(x) if y[i] == 1]
    toxic_comments = [re.sub("[^a-zA-Z\.,!? ]+", "", x) for x in toxic_comments]
    toxic_comments = [re.sub(" +", " ", x) for x in toxic_comments]
    if(max_len != -1):
        toxic_comments = [x for x in toxic_comments if len(x) <= max_len]
    if(max_n != -1):
        toxic_comments = toxic_comments[:max_n]
    return toxic_comments

def get_toxic_data(sample_type,max_n=-1,soft_max_len=-1,max_len=-1,split_sentences=False, use_toxic_classifier=False):
    print('Loading Toxic Samples')
    if(sample_type == 'wiki_toxic'):
        samples = load_wiki_toxic('./data/wiki_toxic/test.csv')
    elif(sample_type in ['abuse_exp', 'abuse_imp', 'off_exp', 'off_imp', 'bad']):
        tweets = get_texts()
        add_imp_exp(tweets)
        samples = sample_tweets(tweets, sample_type)
    elif(sample_type == 'abuse_cls'):
        samples = open('./data/offens_eval/abuse_cls.txt').read().split('\n')
    elif(sample_type == 'abuse_cls_split'):
        samples = open('./data/offens_eval/abuse_cls_split.txt').read().split('\n')
    elif(sample_type == 'abuse_cls_split'):
        samples = open('./data/offens_eval/abuse_cls_split.txt').read().split('\n')
    elif(sample_type == 'abuse_cls_cutoff'):
        samples = open('./data/offens_eval/abuse_exp_cutoff.txt').read().split('\n')
    elif(sample_type == 'abuse_cls_hardcutoff'):
        samples = open('./data/offens_eval/abuse_exp_hardcutoff.txt').read().split('\n')
    elif(sample_type == 'bad_hardcutoff'):
        samples = open('./data/offens_eval/bad_hardcutoff.txt').read().split('\n')
    elif(sample_type == 'abuse_final'):
        samples = open('./data/datasets/abuse_toxic_samples.txt').read().split('\n') #Used for Tdata attack
    elif(sample_type == 'off_final'):
        samples = open('./data/offens_eval/off_toxic_samples.txt').read().split('\n')
    elif(sample_type == 'reddit_final'):
        samples = open('./data/toxic_reddit/reddit_final.txt').read().split('\n')
    elif(sample_type == 'reddit'):
        samples = [x[1] for x in get_toxic_reddit(threshold=0.7, clean_context=False)]
    else:
        print("Invalid sample type!!!")
        exit()
    #Disabled for memory limitations
    if(split_sentences):
        new_samples = []
        for s in samples:
            if(len(s.split()) <= soft_max_len or soft_max_len == -1):
                new_samples.append(s)
            else:
                i = 0
                sents = re.split('(?<=[\.\!\?])\s*', s)
                while(len(' '.join(sents[0:i])) < soft_max_len and i < len(sents)):
                    i += 1
                if(len(' '.join(sents[0:i])) < soft_max_len):
                    print("\n\nFAIL")
                    exit()
                new_samples.append(" ".join(sents[0:i]))
        samples = [s for s in new_samples if (len(s.split()) >= 3 and len(s.split()) <= max_len)]
    else:
        if(max_len != -1):
            samples = [s for s in samples if (len(s.split()) >= 3 and len(s.split()) <= max_len)]
    print(len(samples))
    if(use_toxic_classifier):
        tc = ToxicClassifier(model_type="WTC_bin_prec", device="cuda:0")
        new_samples = []
        for i, s in enumerate(samples):
            print(f"\r{i}/{len(samples)}", end="")
            pred, score = tc.classify_sample(s)
            if(pred == True):
                new_samples.append(s)
        samples = new_samples
    #if(max_len != -1):
    #    samples = [x for x in samples if len(x.split()) <= max_len]
    if(max_n != -1):
        samples = samples[:max_n]
    return samples

def clean_reddit_data():
    top_50 = open('../data/reddit_sample/top_50.txt').read().split('\n')
    top_trigrams = set([x.split('|')[0] for x in open('../data/reddit_sample/trigrams_1000.txt').read().strip().split('\n') if (1000 < int(x.split('|')[1]))])

    f = open('./data/toxic_reddit/BERT_all.txt')
    f2 = open('./data/toxic_reddit/reddit_toxic_filtered.txt' , 'w+')
    i = 0
    print()
    line = f.readline().strip()
    while(line != ''):
        i += 1
        print(f'\r{i}', end='')

        spl = line.split('\t')
        try:
            context, response = spl[0], spl[1]
            if(is_sample_okay(top_trigrams, top_50, context, response)):
                f2.write(line + '\n')
        except:
            print([line])

        line = f.readline().strip()
        line = f.readline().strip()

def classify_reddit_data():
    toxic_classifier = BertForSequenceClassification.from_pretrained('./saves/toxic_classifier/toxic_9.pt').to('cuda:0')
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    k = 0
    f = open('./data/toxic_reddit/reddit_pairs_threshold.txt')
    f2 = open('./data/toxic_reddit/reddit_pairs_context_threshold.txt', 'w+')
    f3 = open('./data/toxic_reddit/errors.txt', 'a+')
    i = 0
    t0 = time.perf_counter()
    line = f.readline().strip()

    while(line != ''):
        i += 1
        #if(i % 10 == k):
        #    continue
        t = time.perf_counter()
        print(f'\r{i+1}/ 6860250 - {(6860250 - i)*(t - t0)/i:.0f} seconds remaining', end='')
        #try:
        response_soft, context, response = line.split('\t')
        context_softs = []
        for s in context.split('|'):
            encoding = bert_tokenizer(s, return_tensors='pt', padding=True, truncation=True).to('cuda:0')
            outputs = toxic_classifier(encoding['input_ids']).logits
            pred = torch.argmax(outputs, axis=1)

            soft = torch.nn.Softmax(dim=1)(outputs)[0][1].item()
            context_softs.append(soft)

        context_softs = [f"{x:.4f}" for x in context_softs]
        f2.write(f'{"|".join(context_softs)}|{response_soft}\t{line}\n')
        #except (RuntimeError, TypeError, NameError, IndexError):
        #    print([line])
        #    f3.write(line + '\n')

        line = f.readline().strip()

def get_bad_words():
    f = open('./data/bad_words.csv')
    lines = f.read().strip().split('\n')
    lines = [x.split(',') for x in lines]
    bad_words = [x[0] for x in lines if x[1] == '1']
    return bad_words

import random as r
def get_toxic_reddit(max_n=-1, threshold=0.5, at_con=False, clean_context=True):
    f = open('./data/datasets/reddit_toxic_testset.txt')
    i = 0
    line = f.readline().strip()
    toxic_samples, safe_samples = [], []
    while(line != ''):
        i += 1
        #if(i == 5875582):
        #    break
        splt = line.split('\t')
        if(len(splt) != 4):
            line = f.readline().strip()
            continue

        if(len(splt[2]) <= 1000 and len(splt[3]) <= 1000):
            if(float(splt[1]) > threshold):
                if(clean_context == False):
                    toxic_samples.append((splt[2], splt[3]))
                elif(all([float(x) < threshold for x in splt[0].split('|')[:-1]])):
                    toxic_samples.append((splt[2], splt[3]))
            elif(at_con):
                safe_samples.append((splt[2], splt[3]))
        if(i % 100 == 0):
            print(f'\r{i}', end='')
        line = f.readline().strip()
    print()
    r.seed(0)
    r.shuffle(toxic_samples)
    r.shuffle(safe_samples)

    if(max_n != -1):
        return toxic_samples[:max_n]
    return toxic_samples

import re
def main():

    #samples = get_toxic_reddit(clean_context=False)
    #print(samples[:5])
    #print(len(samples))
    #exit()

    #f = open('./data/toxic_reddit/reddit_pairs_classified.txt')
    #i = 0
    #line = f.readline()
    #classes = {}
    #while(line != ''):
    #    i += 1
    #    if(i == 5875582):
    #        break
    #    cls = line.split('\t')[0]
    #    classes[cls] = classes.get(cls, 0) + 1
    #    print(f'\r{i}', end='')
    #    line = f.readline()
    #print(classes)
    #exit()



    #6860250
    #f = open('../data/reddit_sample/reddit_pairs_filtered2.txt')
    #f2 = open('../data/reddit_sample/reddit_pairs_filtered2.txt', 'w+')
    #line = f.readline().strip()

    #clean_reddit_data()
    texts = get_toxic_data('reddit', soft_max_len=17, max_len=24, split_sentences=True, use_toxic_classifier=True)
    print(len(texts))
    f = open('./data/toxic_reddit/reddit_final.txt', 'w+')
    f.write('\n'.join(texts))

if(__name__ == '__main__'):
    main()
