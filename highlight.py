import nltk
import spacy
import string
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
import re

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
spacy.cli.download("en_core_web_sm")

nlp = spacy.load("en_core_web_sm")

COMMON_WORDS = {
    "all", "a", "the", "is", "are", "and", "of", "to", "in", "it", "that", "for", "on", "with", "as", "this", "at",
    "by", "from", "an", "be", "has", "was", "or", "not", "but", "which", "we", "you", "they", "their", "our", "can",
    "will", "about", "more", "other", "some", "many", "such", "also", "each", "one", "would", "should", "could", "most",
    "like", "may", "even", "than", "way", "do", "does", "did", "doing", "done", "make", "makes", "made", "using", "use",
    "used", "new", "old", "same", "different", "important", "good", "better", "best", "people", "person", "thing",
    "things", "time", "day", "year", "years", "life", "world", "human", "humans", "system", "systems", "something",
    "anything", "improves", "all", "any"
}

def extract_keywords(text, top_n=10):
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text.lower())
    stopwords = set(nltk.corpus.stopwords.words("english"))

    raw_bigrams = list(ngrams(words, 2))
    bigram_phrases = [" ".join(bigram) for bigram in raw_bigrams]
    bigram_phrases_filtered = [
        bigram for bigram in bigram_phrases
        if not any(word in COMMON_WORDS or word.endswith("ing") for word in bigram.split())
    ]

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(2, 2))
    tfidf_matrix = vectorizer.fit_transform(bigram_phrases_filtered)
    bigram_scores = dict(zip(vectorizer.get_feature_names_out(), tfidf_matrix.sum(axis=0).A1))
    top_bigrams = {bigram for bigram, score in bigram_scores.items() if score >= 0.1}

    words = [word for word in words if word not in stopwords and word not in string.punctuation]
    words = [word for word in words if word not in COMMON_WORDS and not word.endswith("ing")]

    doc = nlp(text)
    named_entities = {ent.text.lower() for ent in doc.ents}
    noun_chunks = {chunk.text.lower() for chunk in doc.noun_chunks}

    vectorizer_single = TfidfVectorizer(stop_words="english", ngram_range=(1, 1), max_features=10)
    tfidf_matrix_single = vectorizer_single.fit_transform([text])
    tfidf_keywords = set(vectorizer_single.get_feature_names_out())

    tagged_words = nltk.pos_tag(words)
    noun_keywords = {word for word, tag in tagged_words if tag in ["NN", "NNS", "NNP", "NNPS"]}

    key_points = top_bigrams | noun_keywords | named_entities | noun_chunks | tfidf_keywords
    key_points = {word.strip(string.punctuation) for word in key_points if len(word) >= 2}

    filtered_key_points = remove_similar_phrases(list(key_points), top_n)
    relevant_sentences = extract_sentences_with_keywords(text, filtered_key_points)

    return sorted(filtered_key_points, key=lambda x: len(x), reverse=True)[:top_n], relevant_sentences

def remove_similar_phrases(phrases, top_n=10, similarity_threshold=0.5):
    from sklearn.metrics.pairwise import cosine_similarity

    if len(phrases) <= 1:
        return phrases

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(phrases)
    cosine_sim = cosine_similarity(tfidf_matrix)

    similar_phrases = set()
    for i in range(len(phrases)):
        for j in range(i + 1, len(phrases)):
            if cosine_sim[i][j] > similarity_threshold:
                similar_phrases.add(phrases[j])

    return [phrase for phrase in phrases if phrase not in similar_phrases][:top_n]

def extract_sentences_with_keywords(text, keywords, top_n=5):
    sentences = nltk.sent_tokenize(text)

    sentence_scores = {}
    for sent in sentences:
        match_count = sum(1 for keyword in keywords if keyword in sent.lower())
        if match_count > 0:
            sentence_scores[sent] = match_count

    sorted_sentences = sorted(sentence_scores.keys(), key=lambda s: sentence_scores[s], reverse=True)
    return sorted_sentences[:top_n]
