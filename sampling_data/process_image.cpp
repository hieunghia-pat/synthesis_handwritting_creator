#include "opencv2/opencv.hpp"

#include <filesystem>
#include <utility>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <cstdlib>

namespace fs = std::filesystem;

class InvalidArguments: public std::exception
{
public:
    InvalidArguments(std::string msg): m_message { msg } {}

    const char* what() const throw()
    {
        return m_message.data();
    }

    ~InvalidArguments() {}

private:
    std::string m_message;
};

cv::Mat merge_img(cv::Mat const& txt_img, cv::Mat const& bg_img)
{
    try {
        float alpha { 0.7 };
        float beta {1 - alpha};

        cv::Size txt_size = txt_img.size();

        cv::Mat tmp_bg_img;
        bg_img(cv::Range(0, txt_size.height), cv::Range(0, txt_size.width)).copyTo(tmp_bg_img);

        cv::Mat result = txt_img*alpha + tmp_bg_img*beta;

        return result;
    }
    catch (std::exception& err)
    {
        std::cout << "Error occured!" << err.what() << std::endl;
        return cv::Mat();
    }
}

int main(int argc, char* argv[])
{
    if (argc == 1)
    {
        throw InvalidArguments("Initial arguments are not correct, required at least 2 argument, but get less than 1 argument.");
        return -1;
    }

    std::vector<std::string> splits { "train_data", "test_data" };
    fs::directory_entry text_dir(argv[1]);
    fs::directory_entry bg_dir(argv[2]);
    fs::recursive_directory_iterator bg_img_files { bg_dir.path() };

    for (std::string const& split: splits)
    {
        fs::recursive_directory_iterator txt_img_files { text_dir.path() / fs::path(split) };
        for (auto const& txt_img_file: txt_img_files)
        {
            cv::Mat txt_img = cv::imread(txt_img_file.path());
            if (txt_img.empty()) continue;

            for (auto const& bg_img_file: bg_img_files)
            {
                cv::Mat bg_img = cv::imread(bg_img_file.path());
                if (bg_img.empty()) continue;

                std::cout << bg_img_file << std::endl;
                std::cout << txt_img_file << std::endl;
                
                cv::Mat final_img = merge_img(txt_img, bg_img);
                cv::imshow("Result", final_img);
                cv::waitKey(100);
            }
        }
    }

    return 0;
}