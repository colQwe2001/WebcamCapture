
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <fstream>
#include <opencv2/dnn.hpp> 
#define NOMINMAX
#include <windows.h> 

const double ALPHA = 0.01;
const cv::Scalar FONT_COLOR(192, 192, 192);
const double FONT_SIZE_BASIC = 0.6;
const int FONT = cv::FONT_HERSHEY_DUPLEX;
const int INFO_PANEL_X = 10;
const int INFO_PANEL_Y = 30;
const int BLUR_SIZE = 5;
const int THRESHOLD_VALUE = 5;
const int MIN_CONTOUR_AREA = 500;
std::chrono::steady_clock::time_point recStartTime;

void ShowInfo(cv::Mat& frame, double smoothedFps, double width, double height) {
	cv::putText(frame,
		"FPS: " + std::to_string(int(smoothedFps)),
		cv::Point(INFO_PANEL_X, INFO_PANEL_Y),
		FONT,
		FONT_SIZE_BASIC,
		FONT_COLOR,
		1, cv::LINE_AA);
	cv::putText(frame,
		"Resolution: " + std::to_string(int(width)) + "x" + std::to_string(int(height)),
		cv::Point(INFO_PANEL_X, INFO_PANEL_Y + 30),
		FONT,
		FONT_SIZE_BASIC,
		FONT_COLOR,
		1, cv::LINE_AA);
}
void ShowREC(cv::Mat& frame, int width) {
	const int REC_X = width - 70;
	const int REC_Y = 25;
	cv::circle(frame, cv::Point(REC_X, REC_Y), 7,
		cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
	cv::putText(frame,
		"REC",
		cv::Point(REC_X + 15, REC_Y + 6),
		FONT,
		FONT_SIZE_BASIC,
		FONT_COLOR,
		1, cv::LINE_AA);
	auto now = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - recStartTime).count();
	int minutes = elapsed / 60;
	int seconds = elapsed % 60;
	cv::putText(frame,
		std::to_string(minutes) + ":" + std::to_string(seconds),
		cv::Point(REC_X + 10, REC_Y + 30),
		FONT,
		FONT_SIZE_BASIC,
		FONT_COLOR,
		1, cv::LINE_AA);
}
void ShowName(cv::Mat& frame, double height, char filename[256]) {
	const int NAME_X = INFO_PANEL_X + 102;
	const int NAME_Y = height - 20;
	cv::putText(frame,
		"Saved as: ",
		cv::Point(INFO_PANEL_X, NAME_Y),
		FONT,
		FONT_SIZE_BASIC,
		FONT_COLOR,
		1, cv::LINE_AA);
	cv::putText(frame,
		filename,
		cv::Point(NAME_X, NAME_Y),
		FONT,
		FONT_SIZE_BASIC,
		FONT_COLOR,
		1, cv::LINE_AA);
}
void DrawLabel(cv::Mat& frame, cv::Rect box, std::vector<std::string> classNames, float confidence, int left, int top, int classId, cv::Scalar color) {
	cv::rectangle(frame, box, color, 2);
	std::string label = classNames[classId] + " " + std::to_string((int)(confidence * 100)) + "%";
	cv::putText(frame, label, cv::Point(left + 2, top + 17), FONT, 0.7, color, 1);
}
void ShowMode(cv::Mat& frame, bool moveMode) {
	std::string text = moveMode ? "Move Mode" : "Detection Mode";
	cv::putText(frame, text, cv::Point(INFO_PANEL_X, INFO_PANEL_Y + 60), FONT, FONT_SIZE_BASIC, FONT_COLOR, 1, cv::LINE_AA);
}

int main()
{
	auto lastBeepTime = std::chrono::steady_clock::now();
	bool Move_mode = false;
	bool Beep_mode = true;
	bool isREC = false;
	char Filename[256];
	char ScreenshotName[256];
	bool IsShowName = false;
	int ShowNameITR = 0;
	bool isMotion = false;
	std::chrono::steady_clock::time_point lastMotionTime;
	auto start = std::chrono::steady_clock::now();
	bool isRecording = false;

	setlocale(LC_ALL, "Russian");
	double smoothedFps = 0.0;
	cv::VideoCapture cap(0);
	cv::VideoWriter writer;

	std::string modelPath = "C:/CPP_Projects/WebcamCapture/x64/Release/models/";
	std::cout << "Файлы найдены, загружаю модель..." << std::endl;
	cv::dnn::Net net = cv::dnn::readNetFromCaffe(
		modelPath + "deploy.prototxt",
		modelPath + "mobilenet_iter_73000.caffemodel"
	);
	std::vector<std::string> classNames;
	std::ifstream ifs(modelPath + "coco.names");
	std::string line;
	while (std::getline(ifs, line)) {
		classNames.push_back(line);
	}
	ifs.close();
	if (cap.isOpened())
	{
		std::cout << "Камера найдена!" << std::endl;
		//вычисление разрешения
		double width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
		double height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
		double camFps = cap.get(cv::CAP_PROP_FPS);
		std::cout << "Разрешение: " << width << "x" << height << std::endl;
		cv::Mat frame;
		int64 prevTick = cv::getTickCount();

		cv::Ptr<cv::BackgroundSubtractor> pBackSub = cv::createBackgroundSubtractorMOG2();
		while (true) {
			cap >> frame;
			cv::Mat blob = cv::dnn::blobFromImage(
				frame,
				0.007843,
				cv::Size(300, 300),  // размер входа сети
				cv::Scalar(127.5, 127.5, 127.5),// вычитание среднего
				false // не менять BGR на RGB
			);
			net.setInput(blob);
			cv::Mat detections = net.forward();
			auto now = std::chrono::steady_clock::now();
			for (int i = 0; i < detections.size[2]; i++) {
				float* data = (float*)detections.ptr<float>(0, 0, i);
				float confidence = data[2];  // уверенность

				if (confidence > 0.5) {
					int classId = (int)data[1];
					bool isInteresting = (classId == 15 || classId == 8 || classId == 12 ||
						classId == 7 || classId == 2 || classId == 4 || classId == 6 || classId == 14 || classId == 19);
					if (!isInteresting) continue;

					// Нормализованные координаты
					int left = (int)(data[3] * frame.cols);
					int top = (int)(data[4] * frame.rows);
					int right = (int)(data[5] * frame.cols);
					int bottom = (int)(data[6] * frame.rows);

					int w = right - left;
					int h = bottom - top;
					cv::Rect box(left, top, w, h);
					if (Move_mode == false) {
						auto sinceLastBeep = std::chrono::duration_cast<std::chrono::seconds>(now - lastBeepTime).count();
						if (classId == 15) {
							DrawLabel(frame, box, classNames, confidence, left, top, classId, cv::Scalar(255, 0, 0));
							if (sinceLastBeep >= 3 && Beep_mode == true) {  // не чаще раза в 3 секунды
								Beep(1500, 300);  // частота 1000 Гц, длительность 200 мс
								lastBeepTime = now;
							}
						}
						else if (classId == 8 || classId == 12) {
							DrawLabel(frame, box, classNames, confidence, left, top, classId, cv::Scalar(0, 255, 0));
							if (sinceLastBeep >= 3 && Beep_mode == true) {  // не чаще раза в 3 секунды
								Beep(1000, 300);  // частота 1000 Гц, длительность 200 мс
								lastBeepTime = now;
							}
						}
						else if (classId == 4 || classId == 6 || classId == 14 || classId == 19 || classId == 7 || classId == 2) {
							DrawLabel(frame, box, classNames, confidence, left, top, classId, cv::Scalar(0, 0, 255));
							if (sinceLastBeep >= 3 && Beep_mode == true) {  // не чаще раза в 3 секунды
								Beep(500, 300);  // частота 1000 Гц, длительность 200 мс
								lastBeepTime = now;
							}
						}
					}
				}
			}
			
			//вычисление фпс
			int64 currTick = cv::getTickCount();
			double frequency = cv::getTickFrequency();
			double fps = 1 / ((currTick - prevTick) / frequency);
			smoothedFps = ALPHA * fps + (1.0 - ALPHA) * smoothedFps;
			prevTick = currTick;

			cv::Mat gray; // черно - белый кадр
			cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY); //преобразуем текущий кадр в черно - белый вариант
			cv::GaussianBlur(gray, gray, cv::Size(BLUR_SIZE, BLUR_SIZE), 0); // сильно размываем черно - белое изображение
				cv::Mat binary; // создаем две матрицы (серая и черно - белая)
				cv::Mat fgMask;
				pBackSub->apply(gray, fgMask);
				cv::threshold(fgMask, binary, 200, 255, cv::THRESH_BINARY);
				std::vector<std::vector<cv::Point>> contours; // создаем векторконтуров
				cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);// обводим контуры изменившихся (движущихся) обьектов
				std::vector<cv::Rect> boxes;
				for (const auto& contour : contours) {// перебираем все полученные контуры
					if (cv::contourArea(contour) > MIN_CONTOUR_AREA) {
						boxes.push_back(cv::boundingRect(contour));
					}
				}
				// фильтр по площади
				std::sort(boxes.begin(), boxes.end(),
					[](const cv::Rect& a, const cv::Rect& b) {
						return a.area() > b.area();
					});
				std::vector<cv::Rect> finalBoxes;
					for (size_t i = 0; i < boxes.size(); i++) {
						bool keep = true;
						for (size_t j = 0; j < finalBoxes.size(); j++) {
							int x1 = std::max(boxes[i].x, finalBoxes[j].x);
							int y1 = std::max(boxes[i].y, finalBoxes[j].y);
							int x2 = std::min(boxes[i].x + boxes[i].width, finalBoxes[j].x + finalBoxes[j].width);
							int y2 = std::min(boxes[i].y + boxes[i].height, finalBoxes[j].y + finalBoxes[j].height);
							int intersectArea = std::max(0, x2 - x1) * std::max(0, y2 - y1);
							double overlap = (double)intersectArea / std::min(boxes[i].area(), finalBoxes[j].area());
							if (overlap > 0.3) {
								keep = false;
								break;
							}
						}
						if (keep) finalBoxes.push_back(boxes[i]);
						}
					if (Move_mode == true) {
						for (const auto& box : finalBoxes) {
							if (box.area() < 5000) {
								cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 2);
							}
							else if (box.area() < 50000) {
								cv::rectangle(frame, box, cv::Scalar(0, 255, 255), 2);
							}
							else {
								cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
							}

						}
					}
						
						
						auto start_waiting = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
						if (!finalBoxes.empty() && start_waiting > 5) {
							// Прямо сейчас есть движение
							lastMotionTime = now;
							isMotion = true;
						}
						else {
							// Движения нет — проверим сколько прошло с последнего
							auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - lastMotionTime).count();
							isMotion = (elapsed < 5);
						}
						if ((isREC && !isMotion) && isRecording == false) {
							writer.release();
							isREC = false;
							IsShowName = true;
							ShowNameITR = 0;
							std::cout << "Запись остановлена (нет движения): " << Filename << std::endl;
						}

			ShowInfo(frame, smoothedFps, width, height);
			ShowMode(frame, Move_mode);

			if (isREC) {
				writer.write(frame);
				ShowREC(frame, width);
			}
			if (IsShowName == true && ShowNameITR < 50) {
				ShowName(frame, height, Filename);
				ShowNameITR++;
			}
			else if (IsShowName == true && ShowNameITR >= 50) {
				IsShowName = false;
				ShowNameITR = 0;
			}
			if (!frame.empty())
			{
				cv::imshow("Video Player", frame);
			}
			else {
				cap.release();
				cv::destroyAllWindows();
				writer.release();
				break;
			}

			int key = cv::waitKey(1);
			struct tm timeinfo;
			bool isPressedREC = (key == 114 || key == 82 || key == 234 || key == 202);
			bool isPressedMode = (key == 109 || key == 77 || key == 252 || key == 220);
			bool isPressedESC = (key == 27);
			bool isPressedScreenshot = (key == 32);
			bool isPressedBeep = (key == 242 || key == 210 || key == 110 || key == 78);
			if (isPressedREC) {
				isRecording = !isRecording;
			}
			if (isPressedESC || cv::getWindowProperty("Video Player", cv::WND_PROP_VISIBLE) < 1) break; //  кнопка ESC (выход)
			if ((isPressedREC && isREC == false) ||(isMotion == true && isREC == false)){
				std::time_t now = std::time(nullptr);
				localtime_s(&timeinfo, &now);
				strftime(Filename, sizeof(Filename),
					"video_%Y_%m_%d_%H-%M-%S.avi", &timeinfo);
				writer.open(Filename,
					cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
					camFps,
					cv::Size(width, height));
				if (!writer.isOpened()) {
					std::cout << "Ошибка: не могу открыть файл для записи!" << std::endl;
					return -1;
				}
				isREC = true;
				recStartTime = std::chrono::steady_clock::now();
			}
			else if (isPressedREC && isREC == true) {
				writer.release();
				isREC = false;
				IsShowName = true;
				ShowNameITR = 0;
				std::cout << "Запись сохранена " << std::endl;
			}
			if (isPressedScreenshot) {
				std::time_t now = std::time(nullptr);
				localtime_s(&timeinfo, &now);
				strftime(ScreenshotName, sizeof(ScreenshotName),
					"video_%Y_%m_%d_%H-%M-%S.png", &timeinfo);
				cv::imwrite(ScreenshotName, frame);
				std::cout << "Скриншот сохранён: " << ScreenshotName << std::endl;
			}
			if (isPressedMode) {
				Move_mode = !Move_mode;
			}
			if (isPressedBeep) {
				Beep_mode = !Beep_mode;
			}
		}
	}
	else {
		std::cout << "Не получается найти камеру! " << std::endl;
	}
	cap.release();
	cv::destroyAllWindows();
	writer.release();
	return 0;
}
