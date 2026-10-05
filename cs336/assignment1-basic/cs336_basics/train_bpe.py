import regex as re
import time
from collections import Counter, defaultdict
from cs336_basics.pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool, cpu_count

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
INIT_NUM = 256


def get_dict(
    input_path: str,
    delimiter_regex: str,
    start_idx: int,
    end_idx: int
) -> Counter[tuple[bytes, ...]]:
    # 임의의 길이를 가지며 모든 요소가 bytes 타입인 튜플
    str_frequency_dict: Counter[str] = Counter()
    with open(input_path, "rb") as f:
        # file 읽기 시작 지점
        f.seek(start_idx)
        # 시작 지점 부터 끝 지점까지 파일 읽고 (바이너리), string으로 변환
        string = f.read(end_idx - start_idx).decode("utf-8", errors="ignore")

        texts: list[str] = []
        # 만약 특수 토큰이 없는 말뭉치가 들어오면, delimiter_regex가 빈 문자열("")이 된다.
        if delimiter_regex == "":
            texts = [string]
        else:
            # [asdfasdasdfasdf, sadfasdfasdgasd, asdfasdf] eos로 나눈 일반텍스트
            # 이렇게 해야 speical token을 살린 상태로 tokenize를 할 수 있다.
            texts = re.split(delimiter_regex, string)
        for text in texts:
            for t in re.finditer(PAT, text):
                str_frequency_dict[t.group()] += 1
    return Counter({
        tuple(bytes([byte]) for byte in bytes(key, "utf-8")): value
        for (key, value) in str_frequency_dict.items()
    })


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    assert len(special_tokens) != 0, "special_tokens must have least one token"
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(INIT_NUM)}
    vocab.update({
        i + INIT_NUM: bytes(special_token, "utf-8")
        for (i, special_token) in enumerate(special_tokens)
    })
    merges: list[tuple[bytes, bytes]] = []
    num_processes = max(1, cpu_count())
    boundaries = []
    # delimiter list, 정규표현식이 인식할 수 있게 escape 처리도 필요함
    delimiter_regex = "|".join([
        re.escape(special_token) for special_token in special_tokens
    ])
    start_time = time.perf_counter()
    with open(input_path, "rb") as f:
        for end_string in special_tokens:
            boundaries.extend(find_chunk_boundaries(
                f, num_processes, bytes(end_string, "utf-8")
            ))
        boundaries = sorted(list(set(boundaries)))
    print("\n1 generate pool time: ", time.perf_counter() - start_time)
    start_time = time.perf_counter()
    with Pool(num_processes) as pool:
        results = pool.starmap(
            get_dict, [(
                    input_path,
                    delimiter_regex,
                    boundaries[i],
                    boundaries[i + 1]
                ) for i in range(len(boundaries) - 1)
            ]
        )
    # 단어가 나온 횟수
    total_freq_dict: Counter[tuple[bytes, ...]] = Counter()
    for result in results:
        total_freq_dict.update(result)
    print("2 pre tokenization: ", time.perf_counter() - start_time)
    start_time = time.perf_counter()
    # 이 byte 쌍이 어떤 단어에서 나온건지
    byte_index: defaultdict[
        tuple[bytes, bytes], set[tuple[bytes, ...]]
    ] = defaultdict(set)
    # 해당 byte 쌍이 전체에서 나온 횟수
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    for (byte_tokens, freq) in total_freq_dict.items():
        for i in range(len(byte_tokens) - 1):
            byte_index[(byte_tokens[i], byte_tokens[i + 1])].add(byte_tokens)
            pair_counts[(byte_tokens[i], byte_tokens[i + 1])] += freq
    if len(vocab) >= vocab_size or len(pair_counts) == 0:
        return vocab, merges
    """
    total_freq_dict
    (t, h, e) × 5
    (t, h, e, n) × 2

    byte_index
    t, h -> (t, h, e), (t, h, e, n)
    h, e -> (t, h, e,), (t, h, e, n)
    e, n -> (t, h, e, n)

    pair_counts
    t, h -> 7
    h. e -> 7
    e, n -> 2
    
    round 1
    max_pair = (t, h)

    old (t, h, e), (t, h, e, n)
    new (th, e), (th, e, n)
    
    

    """
    while len(vocab) < vocab_size:
        if len(pair_counts) == 0:
            break
        # ((b' ', b't'), 641902)
        # 1. pair_counts에서 가장 높은 빈도를 가진 bytes pair 선택
        max_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))
        # 2. 과제 변수에 넣기
        merges.append(max_pair[0])
        merged_byte = max_pair[0][0] + max_pair[0][1]
        vocab[len(vocab)] = merged_byte
        # 3. 병합 해야할 byte pairs를 찾음
        old_merge_targets: set[tuple[bytes, ...]] = byte_index[max_pair[0]].copy()
        for merge_target in old_merge_targets:
            merged_byte_tokens = [] # [th, e]
            i = 0
            while i < len(merge_target):
                if i == len(merge_target) - 1 or (merge_target[i] != max_pair[0][0] or merge_target[i + 1] != max_pair[0][1]):
                    merged_byte_tokens.append(merge_target[i])
                else:
                    merged_byte_tokens.append(merged_byte)
                    i += 1
                i += 1
            merged_byte_tokens = tuple(merged_byte_tokens)
            for i in range(len(merge_target) - 1):  #(t, h, e)
                key = (merge_target[i], merge_target[i + 1])
                pair_counts[key] -= total_freq_dict[merge_target]
                if pair_counts[key] == 0:
                    pair_counts.pop(key)
                byte_index[key].discard(merge_target)
                if len(byte_index[key]) == 0:
                    byte_index.pop(key)
            total_freq_dict[merged_byte_tokens] = total_freq_dict.pop(merge_target)
            for i in range(len(merged_byte_tokens) - 1):
                key = (merged_byte_tokens[i], merged_byte_tokens[i + 1])
                pair_counts[key] += total_freq_dict[merged_byte_tokens]
                byte_index[key].add(merged_byte_tokens)
            
    print("3. merge loop: ", time.perf_counter() - start_time)
    return vocab, merges


if __name__ == "__main__":
    vocab, merges = train_bpe(
        # input_path="./tests/fixtures/corpus.en",
        input_path="./data/TinyStoriesV2-GPT4-valid.txt",
        # input_path="./data/bpe_example.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"]
    )
    # print("vocab")
    # print(vocab)
    # print("merges")
    # print(merges)

"""
("a", "b", "c") 에 ("a", "b")가 있다는걸 어떻게 알지? a, b,를 찾아서 병합해야하는데
지금은 그게 없으니까 모든 쌍을다 순회해야함
심지어 둘이 붙어있는것만 찾아야한단말이지
(a, b, c)를 찾아야지 (a, c, b)는 찾으면 안돼
"""