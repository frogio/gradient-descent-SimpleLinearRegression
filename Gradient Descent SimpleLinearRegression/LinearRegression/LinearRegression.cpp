#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#pragma warning(disable:4996)

#define MAX_DATA						506							// 전체 데이터 개수, M
#define ALPHA							0.01						// 학습률
#define EPOCH							50000						// 학습 횟수

struct Model {
	double w0;
	double w1;
};

struct Target {
	double rm;
	double medv;
};

Target * LoadData();												// BostonHousing Data를 받아온다.
void Training(struct Target * target, struct Model * model);
void Predict(struct Model* model);

void main() {

	struct Target * target = LoadData();
	struct Model model = {1, 1};									// 학습되지 않은 초기 모델

	for (int i = 0; i < MAX_DATA; i++)
		printf("rm, medv : %lf, %lf\n", target[i].rm, target[i].medv);

	for (int i = 0; i < EPOCH; i++)
		Training(target, &model);

	printf("Training Result : %lf * x + %lf * 1\n\n", model.w1, model.w0);

	while (1)
		Predict(&model);


}

void Predict(struct Model* model) {

	double rm;

	printf("Enter what you want to know house price : ");

	scanf("%lf", &rm);

	printf("Predict result : %lf\n", model->w1 * rm + model->w0);
	
}

void Training(struct Target* target, struct Model* model) {			// 경사하강법을 이용한 Training

	Model diff_vec = { 0.f, };

	for (int i = 0; i < MAX_DATA; i++) {
		double predVal = model->w1 * target[i].rm + model->w0 * 1;	// 예측
		double error = predVal - target[i].medv;					// 예측한 값과 실제값의 오차를 구함.
		// 오차 부호 주의!, 반드시 Predict Value - Target Value순서로 빼주어야 함

		diff_vec.w0 += error * 1;
		diff_vec.w1 += error * target[i].rm;

	}

	diff_vec.w0 /= MAX_DATA;
	diff_vec.w1 /= MAX_DATA;
	// 여기까지 손실함수 미분값 계산

	model->w0 -= diff_vec.w0 * ALPHA;
	model->w1 -= diff_vec.w1 * ALPHA;
	// 학습률을 (벡터 이동방향) 곱한 후 더해 가중치 조정

}

Target * LoadData() {
	
	char buf[200] = { 0, };
	struct Target * target;

	target = (struct Target *)malloc(sizeof(Target) * MAX_DATA);

	FILE* fp = fopen("BostonHousing.csv", "rt");
	
	fgets(buf, sizeof(buf), fp);								// csv 파일의 헤드부분을 먼저 읽어온다.

	int tIdx = 0;
	while (fgets(buf, sizeof(buf), fp) != NULL) {				// 데이터를 읽기 시작한다.

		char * data;
		
		for(int i = 0; i < 5; i++){								// 토큰(,) 분리
			data = strchr(buf, ',');
			*data = ' ';
		}

		*strchr(buf, ',') = ' ';
		data++;

		target[tIdx].rm = atof(data);							// rm data 추출
		
		for (int i = 0; i < 7; i++) {							// 토큰(,) 분리
			data = strchr(buf, ',');
			*data = ' ';
		}

		data++;
		target[tIdx].medv = atof(data);							// medv data 추출

		tIdx++;
	}

	fclose(fp);

	return target;

}