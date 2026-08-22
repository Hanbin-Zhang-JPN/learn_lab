from llm_monogatari.vectors import bag_of_words, cosine_similarity, make_vocabulary

sentences = [list("猫が窓で眠る"), list("猫が椅子で眠る"), list("船が港を出る")]
vocabulary = make_vocabulary(sentences)
vectors = [bag_of_words(sentence, vocabulary) for sentence in sentences]

for left, right in [(0, 1), (0, 2), (1, 2)]:
    score = cosine_similarity(vectors[left], vectors[right])
    print(f"{left} vs {right}: {score:.3f}")
