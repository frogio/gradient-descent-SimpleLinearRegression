#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#pragma warning(disable:4996)

#define MAX_DATA						506							// ��ü ������ ����, M
#define ALPHA							0.01						// �н���
#define EPOCH							50000						// �н� Ƚ��

struct Model {
	double w0;
	double w1;
};

struct Target {
	double rm;
	double medv;
};

Target * LoadData();												// BostonHousing Data�� �޾ƿ´�.
void Training(struct Target * target, struct Model * model);
void Predict(struct Model* model);

void main() {

	struct Target * target = LoadData();
	struct Model model = {1, 1};									// �н����� ���� �ʱ� ��

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

void Training(struct Target* target, struct Model* model) {			// ����ϰ����� �̿��� Training

	Model diff_vec = { 0.f, };

	for (int i = 0; i < MAX_DATA; i++) {
		double predVal = model->w1 * target[i].rm + model->w0 * 1;	// ����
		double error = predVal - target[i].medv;					// ������ ���� �������� ������ ����.
		// ���� ��ȣ ����!, �ݵ�� Predict Value - Target Value������ ���־�� ��

		diff_vec.w0 += error * 1;
		diff_vec.w1 += error * target[i].rm;

	}

	diff_vec.w0 /= MAX_DATA;
	diff_vec.w1 /= MAX_DATA;
	// ������� �ս��Լ� �̺а� ���

	model->w0 -= diff_vec.w0 * ALPHA;
	model->w1 -= diff_vec.w1 * ALPHA;
	// �н����� (���� �̵�����) ���� �� ���� ����ġ ����

}

Target * LoadData() {
	
	char buf[200] = { 0, };
	struct Target * target;

	target = (struct Target *)malloc(sizeof(Target) * MAX_DATA);

	FILE* fp = fopen("BostonHousing.csv", "rt");
	
	fgets(buf, sizeof(buf), fp);								// csv ������ ���κ��� ���� �о�´�.

	int tIdx = 0;
	while (fgets(buf, sizeof(buf), fp) != NULL) {				// �����͸� �б� �����Ѵ�.

		char * data;
		
		for(int i = 0; i < 5; i++){								// ��ū(,) �и�
			data = strchr(buf, ',');
			*data = ' ';
		}

		*strchr(buf, ',') = ' ';
		data++;

		target[tIdx].rm = atof(data);							// rm data ����
		
		for (int i = 0; i < 7; i++) {							// ��ū(,) �и�
			data = strchr(buf, ',');
			*data = ' ';
		}

		data++;
		target[tIdx].medv = atof(data);							// medv data ����

		tIdx++;
	}

	fclose(fp);

	return target;

}