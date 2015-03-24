# 모든 문서에 있는 모든 유일한 단어 목록을 생성한다.
def create_vocab_list(data_set):
    # 비어있는 집합 생성
    vocab_set = set([])
    for document in data_set:
        # 연산자 | 는 두 개의 집합 유형의 변수를 합치는 데 사용하는 연산자.
        # 두 개의 집합 통합 생성
        # ex) {'a', 'b', 'c'} | {'c', 'd', 'e'}
        #     -> {'a', 'b', 'c', 'd', 'e'}
        vocab_set = vocab_set | set(document)
    return list(vocab_set)

# 주어진 문서 내에 어휘 목록에 있는 단어가 존재하는지 아닌지를 표현하기 위해 어휘 목록, 문서,
# 1과 0의 출력 벡터를 사용한다.
def set_of_words_2_vec(vocab_list, input_set):
    # 모두 0인 벡터 생성
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print ("the word: %s is not in my Vocablulary!" % word)
    return return_vec

def train(train_matrix, train_category):
    # 문서의 수
    num_train_docs = len(train_matrix)

    # 한문서의 최대 단어수
    num_words = len(train_matrix[0])

    # 폭력적인 문서의 확률 계산
    # 1의 값이 폭력적이기 때문에 train_category를 합하면 폭력적인 문서의 수가 나온다.
    # 폭력적인 문서 수 / 전체 문서 수
    # 현재는 분류 항목이 두개 이기때문에 가능 / 분류 항목이 2개 이상이면 이부분을 수정해야 한다.
    p_abusive = sum(train_category) / float(num_train_docs)

    # 확률 초기화
    # zerors: num_words(인자값) 수 만큼 0. 배열 만듬
    p_0_num = ones(num_words)
    p_1_num = ones(num_words)
    p_0_denom = 2.0
    p_1_denom = 2.0

    # 벡터 추가
    for i in range(num_train_docs):
        if train_category[i] == 1:
            p_1_num += train_matrix[i]
            p_1_denom += sum(train_matrix[i])
        else:
            p_0_num += train_matrix[i]
            p_0_denom += sum(train_matrix[i])

    # 원소 나누기
    p_1_vect = log(p_1_num / p_1_denom)
    p_0_vect = log(p_0_num / p_1_denom)
    return p_0_vect, p_1_vect, p_abusive

def classify(vec_2_classify, p_0_vec, p_1_vec, p_class_1):
    # 원소 곱하기
    # 방식은 두 벡터의 첫 번째 원소들을 곱한 뒤, 두 번째 원소들을 곱하고, 이런식으로 계속해서 끝까지 곱해 가는 방식이다.
    # 그런다음, 어휘집에 있는 모든 단어들에 대한 값을 더하고, 분류 항목의 로그 확률에 더한다.
    #
    p1 = sum(vec_2_classify * p_1_vec) + log(p_class_1)
    p0 = sum(vec_2_classify * p_0_vec) + log(1.0 - p_class_1)

    if p1 > p0:
        return 1
    else:
        return 0

def bag_of_words_2_vec_MN(vocab_list, input_set):
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
    return return_vec