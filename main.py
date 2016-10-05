import sys
import getopt
import argparse
import json
import re
import nltk
import math

from datetime import datetime
from lib import filters

st = nltk.stem.RSLPStemmer() # Portuguese stemmer

######################################################################
# This function strips emoji characters
def filter_emoji(text):
    try:
    # UCS-4
        emoji_pattern = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
    except re.error:
    # UCS-2
        emoji_pattern = re.compile(u'([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])')

    return emoji_pattern.sub('', text)

######################################################################
# This function strips useless terms
def filter_useless(terms):
    useful = []

    pattern = re.compile('^[@#\\d\\r]')

    for term in terms:
        if pattern.match(term): continue

        useful.append(term)

    return useful

######################################################################
# This function stemms terms
def filter_stem(terms):
    #return filter.filter_stemmer(terms)

    stemmed = []

    for term in terms:
        stemmed.append(st.stem(term))

    return stemmed

######################################################################
# This function normalizes a text:
#   - pass to lower case
#   - remove punctiations
#   - remove repetition chars
#   - remove urls
#   - remove accents
def normalize(text):
    text = text.lower()
    text = filters.filter_punct(text)
    text = filters.filter_charRepetition(text)
    text = filters.filter_url(text)
    text = filters.filter_accents(text)
    text = filter_emoji(text)
    text = re.compile('\n').sub('', text)
    return text

######################################################################
# This function tokenizes the text
def tokenize(text):
    return re.compile('[ \n]').split(text)

######################################################################
# This function filter useless terms from a set of terms
def filter(terms, stop_words_file=''):
    terms = filters.filter_small_words(terms, 4)
    terms = filters.filter_numbers(terms)
    #terms = filters.filter_stopwords(terms, use_file=True, stop_words_file=stop_words_file)
    filters.filter_stopwords(terms, stop_words=nltk.corpus.stopwords.words('portuguese'))
    nltk.corpus.stopwords.words('portuguese')
    terms = filter_useless(terms)
    terms = filter_stem(terms)
    return terms

######################################################################
# This function extract relevant features from a list of terms based
# on global it's global count.

######################################################################
# This function counts terms
def count(terms, termCountDict={}, termDocsCountDict={}):
    docTermsCountDict = {}
    termsSet = set()

    for term in terms:
        termsSet.add(term)

        # Local count (within doc)
        if not docTermsCountDict.has_key(term):
            docTermsCountDict[term] = 1
        else:
            docTermsCountDict[term] += 1

        # Global count
        if not termCountDict.has_key(term):
            termCountDict[term] = 1
        else:
            termCountDict[term] += 1

    # Iterate over non duplicated terms and increment docCount of
    # this terms
    for term in termsSet:
        if not termDocsCountDict.has_key(term):
            termDocsCountDict[term] = 1
        else:
            termDocsCountDict[term] += 1

    return docTermsCountDict


######################################################################
# This function collect some stats about terms
def collectStats(terms=[], keywords=[], globalKeywordsCountDict={}):
    #print 'collectStats', terms

    keywordsCountDict = {}

    for keyword in keywords:
        i, keywordHash = 0, " ".join(keyword)

        #print '    keyword', keywordHash

        for term in terms:
            if term == keyword[i]:
                #print '        keyword part found', i
                i += 1
            else:
                i = 0

            # Found this keyword among terms
            if i == len(keyword):
                #print '        keyword total found'

                if not globalKeywordsCountDict.has_key(keywordHash):
                    globalKeywordsCountDict[keywordHash] = 1
                else:
                    globalKeywordsCountDict[keywordHash] += 1

                if not keywordsCountDict.has_key(keywordHash):
                    keywordsCountDict[keywordHash] = 1
                else:
                    keywordsCountDict[keywordHash] += 1

                break

    #print '        keywordsCountDict', keywordsCountDict

    return keywordsCountDict

######################################################################
# This get keywords as a list of strings from keywords file
def getKeywords(keywordsFilePath='keywords.txt'):
    keywords = []

    # Open keywords file
    with open(keywordsFilePath) as keywordsFile:
        for keyword in keywordsFile:
            keyword = normalize(unicode(keyword, "utf-8"))
            keyword = keyword.split(' ')
            keywords.append(keyword)

    return keywords

######################################################################
# This function evaluates tfidf of terms in a document
#     - docTermsCountDict : dict with terms count of a document
#     - termDocsCountDict: dict with doc count of terms over corpora
#     - numDocs : total number of docs in corpora
def tfidf(docTermsCountDict={}, termDocsCountDict={}, numDocs=0):
    tfidfDict = {}

    for term, count in docTermsCountDict.iteritems():
        tfidfDict[term] = count * math.log(numDocs/termDocsCountDict[term])

    return tfidfDict


######################################################################
# Main function
def main():
    # Parse command line options
    parser = argparse.ArgumentParser(
        prog='PROG',
        usage='%(prog)s [options]',
        description='BDA analizer'
    )

    parser.add_argument(
        '-i',
        '--input',
        default='input.json',
        help='input file path'
    )

    parser.add_argument(
        '-o',
        '--output',
        default='output.json',
        help='output file path'
    )

    parser.add_argument(
        '-s',
        '--stopwords',
        default='stopwords.txt',
        help='stop words file path'
    )

    parser.add_argument(
        '-k',
        '--keywords',
        default='keywords.txt',
        help='key words file path'
    )

    parser.add_argument(
        '-m',
        '--max',
        type=int,
        default=-1,
        help='max number of documents to analize (-1 to all)'
    )

    parser.add_argument(
        '-c',
        '--count',
        default='count.json',
        help='count terms json file'
    )

    parser.add_argument(
        '-t',
        '--stats',
        default='stats.json',
        help='stats json file'
    )

    parser.add_argument(
        '-d',
        '--doc',
        default='docs.json',
        help='docs info json file'
    )

    args = parser.parse_args()
    keywords = getKeywords(args.keywords)

    #print 'keywords', keywords
    #cprint '\n'

    # Stats dicts
    termsCountDict, termDocsCountDict, globalKeywordsCountDict = {},{},{}

    # Iterate over lines in input file
    with open(args.input) as input:
        numDocs = 0
        docs = []    # Store doc relevant info to be inserted into mongodb

        for line in input:
            if args.max > 0 and numDocs >= args.max: break

            twitterDoc = json.loads(line)

            # Relevant twitter doc info to be inserted into mongodb
            doc = {
                'originalText': twitterDoc['text'],
                'createdAt': datetime.fromtimestamp(twitterDoc['created_at']['$date']/1000).isoformat()
            }

            # Get text to be processed
            text = twitterDoc['text']

            # Normalize text
            text = normalize(text)
            doc['text'] = text

            # Tokenize text
            tokens = tokenize(text)
            #print 'tokens', tokens

            # Collect stats over terms
            doc['keywords'] = list(collectStats(tokens, keywords, globalKeywordsCountDict).keys())

            # Extract filter useless terms
            terms = filter(tokens, stop_words_file=args.stopwords)
            #doc['terms'] = list(set(terms))

            # Count terms globally
            doc['termsCount'] = count(terms, termsCountDict, termDocsCountDict)

            # Push doc into docs list
            docs.append(doc)

            numDocs += 1

        #print 'terms', terms
        #print 'count', countDict

        # Extract features (relevant terms)
        #features = extract(termsCountDict, termDocsCountDict)

        # Extract features
        sortedTermsByCount = sorted(termsCountDict, key=termsCountDict.__getitem__)
        sortedTermsCount = []

        for term in sortedTermsByCount[::-1]:
            sortedTermsCount.append("%s:%s" % (term, termsCountDict[term]))

        # Evaluate tfidf for each collected doc
        for doc in docs:
            doc['tfidf'] = tfidf(doc['termsCount'], termDocsCountDict, numDocs)

        # Save features
        with open('sorted_count.json', 'w') as f:
            f.write(json.dumps(sortedTermsCount, indent=4))

        # Save docs count to file
        with open('docs_count.json', 'w') as f:
            f.write(json.dumps(termDocsCountDict, indent=4, sort_keys=True))

        # Save count to file
        with open(args.doc, 'w') as f:
            f.write(json.dumps(docs, indent=4, sort_keys=True))

        # Save count to file
        with open(args.count, 'w') as f:
            f.write(json.dumps(termsCountDict, indent=4, sort_keys=True))

        with open(args.stats, 'w') as f:
            f.write(json.dumps(globalKeywordsCountDict, indent=4, sort_keys=True))

        print "numDocs =", numDocs

######################################################################
# Call main function
if __name__ == "__main__":
    main()
