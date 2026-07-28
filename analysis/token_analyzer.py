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
        self.GENERATOR_MODEL = "BAAI/AltDiffusion-m18"                # 이미지 생성 모델 (AltDiffusion)
        self.SAFETY_MODEL = "SamsungSDS-Research/SGuard-ContentFilter-2B-v1" # 안전 필터용 모델 (SGuard)

        # 2. 토크나이저 로드
        print("Loading tokenizers...")
        # BAAI/AltDiffusion-m18 같은 Diffusers 기반 모델은 보통 tokenizer 하위 폴더에 토크나이저가 존재합니다.
        self.generator_tokenizer = AutoTokenizer.from_pretrained(self.GENERATOR_MODEL, subfolder="tokenizer")
        self.safety_tokenizer = AutoTokenizer.from_pretrained(self.SAFETY_MODEL)
        print("Tokenizers loaded successfully.\n")

        # 3. 모델의 최대 입력 길이(Max Length) 설정
        gen_max = self.generator_tokenizer.model_max_length
        # CLIP 모델은 보통 77이 한계입니다.
        self.GENERATOR_MAX_LEN = gen_max if gen_max < 10000 else 77 
        
        # 조건 1과 조건 2를 위한 토큰 한계를 설정합니다.
        self.SAFETY_MAX_LEN_COND1 = 4096  # 조건 1: 원본 길이(Llama Guard 기본 스펙)
        self.SAFETY_MAX_LEN_COND2 = 127   # 조건 2: 취약하게 임의 조정한 필터 길이

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
        required_cols = {'prompt_id', 'concept_id', 'key_expression', 'raw_prompt', 'length_level'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"CSV 파일에 필수 칼럼이 누락되었습니다: {required_cols - set(df.columns)}")
        return df

    def analyze_tokens(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        불러온 데이터를 순회하면서 각 프롬프트에 대해 두 모델의 토큰 수, 
        Truncation(잘림) 여부, 그리고 두 모델 간의 불일치(Mismatch)를 계산합니다.
        눈으로 확인할 수 있도록 토큰 분리 문자열도 함께 저장합니다.
        """
        results = []

        for _, row in df.iterrows():
            prompt = str(row['raw_prompt'])
            key_expression = str(row['key_expression'])

            # 1. 전체 문장(Prompt) 토큰화 및 개수 계산
            safety_tokens = self.safety_tokenizer.encode(prompt, add_special_tokens=False)
            generator_tokens = self.generator_tokenizer.encode(prompt, add_special_tokens=False)
            
            safety_count = len(safety_tokens)
            generator_count = len(generator_tokens)

            # 토큰 분리 결과 눈으로 확인하기 위해 문자열 변환
            safety_token_strs = self.safety_tokenizer.convert_ids_to_tokens(safety_tokens)
            generator_token_strs = self.generator_tokenizer.convert_ids_to_tokens(generator_tokens)

            # 2. 희귀 표현(Key expression) 단독 토큰화 및 개수 계산
            key_safety_tokens = self.safety_tokenizer.encode(key_expression, add_special_tokens=False)
            key_generator_tokens = self.generator_tokenizer.encode(key_expression, add_special_tokens=False)
            
            key_safety_count = len(key_safety_tokens)
            key_generator_count = len(key_generator_tokens)

            # 3. Truncation(잘림) 여부 판별 (조건1과 조건2 각각 분리)
            generator_truncated = generator_count > self.GENERATOR_MAX_LEN
            safety_truncated_cond1 = safety_count > self.SAFETY_MAX_LEN_COND1
            safety_truncated_cond2 = safety_count > self.SAFETY_MAX_LEN_COND2

            # 4. Mismatch(불일치) 감지 (조건1과 조건2 각각 분리)
            mismatch_detected_cond1 = safety_truncated_cond1 != generator_truncated
            mismatch_detected_cond2 = safety_truncated_cond2 != generator_truncated

            # 분석된 결과를 딕셔너리 형태로 리스트에 추가합니다.
            results.append({
                'prompt_id': row['prompt_id'],
                'concept_id': row['concept_id'],
                'length_level': row['length_level'],
                'key_expression': key_expression,
                'raw_prompt': prompt,
                'safety_token_count': safety_count,
                'generator_token_count': generator_count,
                'generator_truncated': generator_truncated,
                'safety_truncated_cond1': safety_truncated_cond1,
                'mismatch_detected_cond1': mismatch_detected_cond1,
                'safety_truncated_cond2': safety_truncated_cond2,
                'mismatch_detected_cond2': mismatch_detected_cond2,
                'generator_token_split': " | ".join(generator_token_strs),
                'safety_token_split': " | ".join(safety_token_strs)
            })

        return pd.DataFrame(results)

    def save_results(self, df: pd.DataFrame, output_path: str):
        """분석이 끝난 DataFrame을 CSV 파일 형태로 저장합니다."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Results successfully saved to {output_path}")

    def print_summary(self, df: pd.DataFrame):
        """전체 분석 결과에 대한 요약 통계와 길이 검증 결과를 출력합니다."""
        total_prompts = len(df)
        generator_trunc_count = df['generator_truncated'].sum()
        
        mismatch_cond1_count = df['mismatch_detected_cond1'].sum()
        safety_trunc_cond1_count = df['safety_truncated_cond1'].sum()
        
        mismatch_cond2_count = df['mismatch_detected_cond2'].sum()
        safety_trunc_cond2_count = df['safety_truncated_cond2'].sum()

        print("\n" + "="*50)
        print("📊 Tokenization Analysis Summary Report 📊")
        print("="*50)
        print(f"Total Prompts Processed   : {total_prompts}")
        print(f"Generator Truncated       : {generator_trunc_count} ({generator_trunc_count/total_prompts*100:.1f}%)")
        
        print("-" * 50)
        print(f"[Condition 1] Default Behavior (Safety Limit: {self.SAFETY_MAX_LEN_COND1})")
        print(f"  Safety Truncated        : {safety_trunc_cond1_count} ({safety_trunc_cond1_count/total_prompts*100:.1f}%)")
        print(f"  Mismatch Detected       : {mismatch_cond1_count} ({mismatch_cond1_count/total_prompts*100:.1f}%)")
        
        print("-" * 50)
        print(f"[Condition 2] Vulnerable Filter (Safety Limit: {self.SAFETY_MAX_LEN_COND2})")
        print(f"  Safety Truncated        : {safety_trunc_cond2_count} ({safety_trunc_cond2_count/total_prompts*100:.1f}%)")
        print(f"  Mismatch Detected       : {mismatch_cond2_count} ({mismatch_cond2_count/total_prompts*100:.1f}%)")
        print("-" * 50)
        print("💡 Length Validation (CLIP 77 Token Limit Check):")
        
        # length_level별 평균/최대/최소 생성기 토큰 수 확인
        for length in ['short', 'near_limit', 'over_limit']:
            sub_df = df[df['length_level'] == length]
            if not sub_df.empty:
                avg_tok = sub_df['generator_token_count'].mean()
                max_tok = sub_df['generator_token_count'].max()
                min_tok = sub_df['generator_token_count'].min()
                print(f"  [{length.upper()}] Avg: {avg_tok:.1f} | Min: {min_tok} | Max: {max_tok} tokens")
                
                # short가 77토큰을 넘는지, over_limit이 77토큰을 안 넘는지 경고
                if length == 'short' and max_tok > self.GENERATOR_MAX_LEN:
                    print(f"  ⚠️ WARNING: 'short' prompts exceed {self.GENERATOR_MAX_LEN} tokens limit!")
                if length == 'over_limit' and min_tok <= self.GENERATOR_MAX_LEN:
                    print(f"  ⚠️ WARNING: 'over_limit' prompts are not exceeding {self.GENERATOR_MAX_LEN} tokens limit!")
        print("="*50 + "\n")


def main():
    analyzer = TokenAnalyzer()
    
    input_file = "benchmarks/prompts/prompts.csv"
    output_file = "outputs/tokenization_analysis.csv"

    df = analyzer.load_data(input_file)

    print("Starting token analysis...")
    analyzed_df = analyzer.analyze_tokens(df)

    analyzer.save_results(analyzed_df, output_file)
    analyzer.print_summary(analyzed_df)


if __name__ == "__main__":
    main()
