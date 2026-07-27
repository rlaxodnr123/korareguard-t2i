import os
import pandas as pd
from transformers import AutoTokenizer

class TokenAnalyzer:
    """
    이미지 생성 모델(Generator)과 텍스트 안전 필터(Safety Filter) 간의 
    토큰화(Tokenization) 불일치 및 텍스트 잘림(Truncation) 현상을 분석하는 클래스입니다.
    """

    def __init__(self):
        # 1. 사용할 토크나이저 모델 이름 정의
        self.GENERATOR_MODEL = "openai/clip-vit-base-patch32" # 이미지 생성 모델 (Stable Diffusion 등에서 주로 사용)
        self.SAFETY_MODEL = "bert-base-multilingual-cased"     # 안전 필터용 다국어 모델

        # 2. 토크나이저 로드
        print("Loading tokenizers...")
        self.generator_tokenizer = AutoTokenizer.from_pretrained(self.GENERATOR_MODEL)
        self.safety_tokenizer = AutoTokenizer.from_pretrained(self.SAFETY_MODEL)
        print("Tokenizers loaded successfully.\n")

        # 3. 모델의 최대 입력 길이(Max Length) 설정
        # 질문해주신 것처럼 하드코딩(직접 지정)하는 대신, HuggingFace 토크나이저 객체에 내장된 
        # model_max_length 속성을 동적으로 가져오도록 수정했습니다. 
        # (단, 일부 모델은 1e30 처럼 비정상적인 최대길이가 설정되어 있을 수 있어 예외처리(Fallback)를 추가했습니다.)
        
        gen_max = self.generator_tokenizer.model_max_length
        # CLIP 모델은 보통 77이 한계입니다.
        self.GENERATOR_MAX_LEN = gen_max if gen_max < 10000 else 77 
        
        safe_max = self.safety_tokenizer.model_max_length
        # BERT 계열은 보통 512지만 요구사항에 맞춰 128로 가정하거나 모델 기본값을 씁니다.
        self.SAFETY_MAX_LEN = safe_max if safe_max < 10000 else 128 

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        분석할 프롬프트 데이터(CSV)를 불러옵니다.
        파일이 존재하지 않으면 FileNotFoundError가 발생합니다.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"필수 입력 데이터 파일이 '{file_path}' 경로에 존재하지 않습니다.")
            
        print(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        
        # 필수 칼럼이 데이터프레임에 모두 존재하는지 확인
        required_cols = {'prompt_id', 'concept_id', 'expression_type', 'key_phrase', 'raw_prompt'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"CSV 파일에 필수 칼럼이 누락되었습니다: {required_cols - set(df.columns)}")
        return df

    def analyze_tokens(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        불러온 데이터를 순회하면서 각 프롬프트에 대해 두 모델의 토큰 수, 
        Truncation(잘림) 여부, 그리고 두 모델 간의 불일치(Mismatch)를 계산합니다.
        """
        results = []

        for _, row in df.iterrows():
            prompt = str(row['raw_prompt'])
            key_phrase = str(row['key_phrase'])

            # 1. 전체 문장(Prompt) 토큰화 및 개수 계산
            # add_special_tokens=False를 주어 순수하게 텍스트가 몇 개의 토큰으로 쪼개지는지만 셉니다.
            safety_tokens = self.safety_tokenizer.encode(prompt, add_special_tokens=False)
            generator_tokens = self.generator_tokenizer.encode(prompt, add_special_tokens=False)
            
            safety_count = len(safety_tokens)
            generator_count = len(generator_tokens)

            # 2. 희귀 표현(Key phrase) 단독 토큰화 및 개수 계산
            # 특정 단어가 각 모델에서 얼마나 과도하게 쪼개지는지(Subword Tokenization) 파악합니다.
            key_safety_tokens = self.safety_tokenizer.encode(key_phrase, add_special_tokens=False)
            key_generator_tokens = self.generator_tokenizer.encode(key_phrase, add_special_tokens=False)
            
            key_safety_count = len(key_safety_tokens)
            key_generator_count = len(key_generator_tokens)

            # 3. Truncation(잘림) 여부 판별
            # 토큰 개수가 각 모델의 최대 한계(MAX_LEN)를 초과했는지 True/False로 저장합니다.
            safety_truncated = safety_count > self.SAFETY_MAX_LEN
            generator_truncated = generator_count > self.GENERATOR_MAX_LEN

            # 4. Mismatch(불일치) 감지
            # 한 모델에서는 문장이 잘렸는데 다른 모델에서는 안 잘린 경우(True != False) 불일치로 간주합니다.
            mismatch_detected = safety_truncated != generator_truncated

            # 분석된 결과를 딕셔너리 형태로 리스트에 추가합니다.
            results.append({
                'prompt_id': row['prompt_id'],
                'concept_id': row['concept_id'],
                'expression_type': row['expression_type'],
                'key_phrase': key_phrase,
                'raw_prompt': prompt,
                'safety_token_count': safety_count,        # Safety 필터 전체 토큰 수
                'generator_token_count': generator_count,  # 생성 모델 전체 토큰 수
                'key_phrase_safety_tokens': key_safety_count,      # 희귀 표현이 Safety 필터에서 쪼개진 수
                'key_phrase_generator_tokens': key_generator_count,# 희귀 표현이 생성 모델에서 쪼개진 수
                'safety_truncated': safety_truncated,      # Safety 필터 최대 길이 초과 여부
                'generator_truncated': generator_truncated,# 생성 모델 최대 길이 초과 여부
                'mismatch_detected': mismatch_detected     # 두 모델의 Truncation 결과가 다른지 여부
            })

        return pd.DataFrame(results) # 딕셔너리 리스트를 다시 DataFrame(표 형태)으로 변환하여 반환

    def save_results(self, df: pd.DataFrame, output_path: str):
        """분석이 끝난 DataFrame을 CSV 파일 형태로 저장합니다."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True) # 폴더가 없으면 자동 생성
        df.to_csv(output_path, index=False)
        print(f"Results successfully saved to {output_path}")

    def print_summary(self, df: pd.DataFrame):
        """전체 분석 결과에 대한 요약 통계를 콘솔에 보기 좋게 출력합니다."""
        total_prompts = len(df)
        mismatch_count = df['mismatch_detected'].sum()
        safety_trunc_count = df['safety_truncated'].sum()
        generator_trunc_count = df['generator_truncated'].sum()

        print("\n" + "="*50)
        print("📊 Tokenization Analysis Summary Report 📊")
        print("="*50)
        print(f"Total Prompts Processed   : {total_prompts}")
        print(f"Mismatch Detected         : {mismatch_count} ({mismatch_count/total_prompts*100:.1f}%)")
        print(f"Safety Truncated          : {safety_trunc_count} ({safety_trunc_count/total_prompts*100:.1f}%)")
        print(f"Generator Truncated       : {generator_trunc_count} ({generator_trunc_count/total_prompts*100:.1f}%)")
        print("="*50 + "\n")


def main():
    # 1. 분석기 객체 생성
    analyzer = TokenAnalyzer()
    
    # 2. 입출력 경로 지정
    input_file = "data/prompts.csv"
    output_file = "outputs/tokenization_analysis.csv"

    # 3. 데이터 로딩 (파일 없으면 더미데이터 작동)
    df = analyzer.load_data(input_file)

    # 4. 토큰화 불일치 분석 실행
    print("Starting token analysis...")
    analyzed_df = analyzer.analyze_tokens(df)

    # 5. 결과 파일 저장
    analyzer.save_results(analyzed_df, output_file)

    # 6. 통계 리포트 출력
    analyzer.print_summary(analyzed_df)


if __name__ == "__main__":
    main()
