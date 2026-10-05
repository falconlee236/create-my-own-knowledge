### 2. BPE 학습 전체 파이프라인에서 채워야 할 핵심 단계들                                        
                                                                                                   
  현재 코드는 **"작업 구간을 나누어 워커 프로세스들에 던져주는 구조"**까지 작성된 상태입니다. BPE를
  완성하려면 다음 단계들을 순차적으로 구현해야 합니다:                                             
                                                                                                   
  #### ① 사전 토큰화 (Pre-tokenization - get_dict 내부)                                            
                                                                                                   
  • 워커 프로세스가 파일의 할당된 바이트 구간을 읽어 들입니다.                                     
  • GPT-2 정규표현식(regex 모듈 사용)을 이용해 텍스트를 단어 단위(pre-tokens)로 쪼갭니다.          
  • 각 단어를 바이트 시퀀스로 표현하고, 해당 청크 내에서의 **단어 출현 빈도수                      
  딕셔너리(dict[tuple[bytes, ...], int])**를 만들어 반환합니다.                                    
                                                                                                   
  #### ② 빈도수 병합 (Aggregation)                                                                 
                                                                                                   
  • 메인 프로세스에서 results로 모인 각 워커의 딕셔너리들을 순회하며, 전체 말뭉치에 대한 통합 단어 
  빈도수 딕셔너리를 만듭니다.                                                                      
                                                                                                   
  #### ③ 초기 어휘 사전(Vocabulary) 구성                                                           
                                                                                                   
  • 0번부터 255번까지의 기본 256개 바이트 토큰을 기본 어휘로 등록합니다.                           
  • special_tokens를 어휘 사전에 추가합니다. (특수 토큰의 ID 부여 순서나 어휘 등록 방식에 대한 과제
  PDF의 설명을 꼭 확인하세요.)                                                                     
                                                                                                   
  #### ④ BPE 반복 병합 루프 (Merge Loop)                                                           
  
  • 목표 어휘 크기(vocab_size)에 도달할 때까지 반복:
      1. 현재 단어 딕셔너리에서 가장 자주 인접하여 등장하는 바이트 쌍(pair)을 찾습니다.            
      2. 동점 처리(Tie-breaking): 빈도수가 같은 쌍이 여러 개일 경우의 우선순위 규칙(보통           
      사전순/lexicographical order)을 과제 PDF에서 확인하여 일치시킵니다.
      3. 가장 빈번한 쌍을 병합(merges에 추가, 새 토큰을 vocab에 추가).
      4. 단어 딕셔너리 내부의 해당 바이트 쌍들을 새 토큰으로 치환(업데이트).
  
  ──────
  ### 3. 테스트 대비 사전 팁 (test_train_bpe_speed)
  
  • tests/test_train_bpe.py의 속도 테스트를 보면, 작은 말뭉치(corpus.en)에 대해 1.5초 이내에 BPE   
  학습을 마쳐야 합니다.
  • 파일 크기가 매우 작은 경우, 멀티프로세스를 띄우는 오버헤드(프로세스 fork/spawn 비용)가 실제    
  연산 시간보다 훨씬 커질 수 있습니다.
  • 파일 크기가 너무 작을 때의 처리나, 병합 루프에서의 자료구조 효율성(매 반복마다 전체 데이터를   
  처음부터 재스캔하지 않는 방법 등)을 염두에 두고 구현을 진행하시면 좋습니다.


  ### 다음 단계: 메인 프로세스 작업 이어가기
  
  pool.starmap이 끝나고 나면 results에는 각 프로세스가 반환한 Counter 객체들이 들어 있습니다. 이제 아래 순서대로 BPE 알고리즘을 이어나가시면 됩니다:  
  
  1. 통합 빈도수 카운터 생성 (Aggregation)
      • 빈 Counter를 하나 만들고, results 안의 각 프로세스 결과들을 순회하며 하나로 합산합니다. (Counter.update() 메서드를 활용하면 아주 편리합니다.) 
  2. 초기 어휘 사전(vocab) 구성
      • 0번부터 255번까지의 정수 ID를 단일 바이트(bytes([i]))에 매핑하여 256개의 기본 어휘를 만듭니다.
      • special_tokens도 어휘 사전에 추가합니다. (과제 PDF 7페이지와 9페이지를 참고하여 특수 토큰이 어떤 순서/ID로 어휘집에 추가되는지 확인해 보세요.)
  3. BPE 반복 병합 루프 (Merges)
      • 어휘 사전 크기가 vocab_size가 될 때까지 다음 과정을 반복합니다:
          • 현재 단어들의 각 인접 바이트 쌍의 출현 빈도를 집계합니다.
          • 가장 빈도수가 높은 쌍을 선택합니다 (동점일 경우 과제 명세에 따른 사전식 순서 타이 브레이킹 적용).
          • 선택된 쌍을 merges에 추가하고, 새로 만들어진 결합 토큰을 vocab에 등록합니다.
          • 단어들 안에서 해당 쌍을 새로운 토큰으로 치환하여 사전을 업데이트합니다.


(cs336-basics) ☁  assignment1-basics [main] ⚡  uv run python -m http.server 8000
https://lsy-code.dev-lr.com/proxy/8000/
Serving HTTP on 0.0.0.0 port 8000 (http://0.0.0.0:8000/) ...
127.0.0.1 - - [03/Oct/2026 21:16:30] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [03/Oct/2026 21:16:41] "GET /profile.svg HTTP/1.1" 200 -
127.0.0.1 - - [03/Oct/2026 21:19:11] "GET /profile.svg?x=0&y=52 HTTP/1.1" 200 -

py-spy record -o profile.svg -- python cs336_basics/train_bpe.py
uv run pytest tests/test_train_bpe.py
uv run pytest tests/test_train_bpe.py -s