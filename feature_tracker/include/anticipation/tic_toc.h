#pragma once

#include <ctime>
#include <cstdlib>
#include <chrono>
#include <string>
#include <fstream>

class TicToc
{
public:
TicToc(const std::string& file) : file_(file)
{
	log_.open(file + ".bin", std::ios::binary);
}

TicToc()
{
	tic();
}

~TicToc() = default;


void tic() {
	start = std::chrono::system_clock::now();
}

double toc()
{
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;

  double ms = elapsed_seconds.count() * 1000;

  if (log_.is_open()) {
	  log_.write(reinterpret_cast<char *>(&ms), sizeof ms);
		log_.flush();
	}

  return ms;
}

private:
  std::chrono::time_point<std::chrono::system_clock> start, end;

  std::string file_;
  std::ofstream log_;
};
