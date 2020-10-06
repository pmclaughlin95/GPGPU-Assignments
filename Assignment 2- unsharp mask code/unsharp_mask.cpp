#include "unsharp_mask.hpp"
#include "ppm.hpp"
#include <chrono>
#include "unsharp_mask.hpp"
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include "util.hpp"
#include "err_code.hpp"

#ifndef KERNEL_PATH
#define KERNEL_PATH "../unsharp_mask.cl"
#endif

#define NANOSECONDS_TO_SECONDS 1.0e-9f


// GPGPU Assignment 2
// Unsharp Mask
// B00268411 Paul McLaughlin
// Apply an unsharp mask to the 24-bit PPM loaded from the file path of
// the first input argument; then write the sharpened output to the file path
// of the second argument. The third argument provides the blur radius.


 
void deviceInfo() {
	std::vector<cl::Platform> platforms;
	//checks for available platforms
  	cl::Platform::get(&platforms);
	//selects a platform to use
	cl::Platform platform = platforms[0];
	
	//Displays the device information
	std::cout << "\n-------------------------" << std::endl;
  	std::cout << "Platform being used: ";
	std::cout << "\n-------------------------\n";

	std::string s;
	platform.getInfo(CL_PLATFORM_NAME, &s);
	std::cout << "Platform: " << s << std::endl;

	platform.getInfo(CL_PLATFORM_VENDOR, &s);
	std::cout << "\tVendor:  " << s << std::endl;
	 
	platform.getInfo(CL_PLATFORM_VERSION, &s);
	std::cout << "\tVersion: " << s << std::endl;

	std::cout << "\n-------------------------\n";

}

void unsharp_mask_cl(unsigned char* out, const unsigned char* in, const int blur_radius,
	const unsigned w, const unsigned h, const unsigned nchannels)
{
	std::vector<cl::Platform> platforms;
	//checks for available platforms
	cl::Platform::get(&platforms);
	//selects platform to use
	cl::Platform platform = platforms[0];


	std::vector<cl::Device> devices;
	//checs or availaable devices
	platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
	//selects a device to use
	cl::Device device = devices[0];

	//create context
	cl::Context context(device);
	cl::Program program(context, util::loadProgram(KERNEL_PATH), true);
	cl::CommandQueue queue(context);
	//Defining unsharp mask operation variables
	int offset = 0;
	const int size = w * h * nchannels;
	const auto alpha = 1.5f; const auto beta = -0.5f; const auto gamma = 0.0f;

		//Create the blur kernel
		cl::Kernel blur(program, "blur");
		
		//host and image data to be defined and written
		cl::Buffer d_in(context, CL_MEM_READ_WRITE, size);
		cl::Buffer d_blur1(context, CL_MEM_READ_WRITE, size);
		cl::Buffer d_blur2(context, CL_MEM_READ_WRITE, size);
		cl::Buffer d_blur3(context, CL_MEM_READ_WRITE, size);
		cl::Buffer d_out(context, CL_MEM_READ_WRITE, size);
		queue.enqueueWriteBuffer(d_in, true, offset, size, in);


		//arguments passed into the blur kernel
		blur.setArg(0, d_blur1);
		blur.setArg(1, d_in);
		blur.setArg(2, blur_radius);
		blur.setArg(3, w);
		blur.setArg(4, h);
		blur.setArg(5, nchannels);

		//Add the blur kernel to the queue and run it
		queue.enqueueNDRangeKernel(blur, cl::NullRange, cl::NDRange(w, h), cl::NullRange, NULL);

		//Repeat the process for the second and third blur operations...
		blur.setArg(0, d_blur2);
		blur.setArg(1, d_blur1);
		queue.enqueueNDRangeKernel(blur, cl::NullRange, cl::NDRange(w, h), cl::NullRange, NULL);
		blur.setArg(0, d_blur3);
		blur.setArg(1, d_blur2);
		queue.enqueueNDRangeKernel(blur, cl::NullRange, cl::NDRange(w, h), cl::NullRange, NULL);

		//Create the add_weighted kernel
		cl::Kernel add_weighted(program, "add_weighted");

		//Arguments passed into the add_weighted kernel
		add_weighted.setArg(0, d_out);
		add_weighted.setArg(1, d_in);
		add_weighted.setArg(2, d_blur3);
		add_weighted.setArg(3, w);
		add_weighted.setArg(4, h);
		add_weighted.setArg(5, alpha);
		add_weighted.setArg(6, beta);
		add_weighted.setArg(7, gamma); 
		add_weighted.setArg(8, nchannels);

		//Enqueue and run the add_weighted kernel
		queue.enqueueNDRangeKernel(add_weighted, cl::NullRange, cl::NDRange(w, h), cl::NullRange, NULL);

		//Data is read from the buffer back to the host
		queue.enqueueReadBuffer(d_out, true, 0, size, out);
		//queue is finished
		queue.finish();
	
}

	int main(int argc, char* argv[])
	{
		// variables for the image loading, reading and writing and for the unsharp mask operation.
		const char* ifilename = argc > 1 ? argv[1] : "../ghost-town-8k.ppm";
		const char* ofilename = argc > 2 ? argv[2] : "../out.ppm";
		const int blur_radius = 5;
		deviceInfo();
		ppm img;
		std::vector<unsigned char> data_in, data_sharp;
 		std::cout << "\nLoading Image...\n";
		std::cout << "\n-------------------------\n";

		// read image file that is to have the unsharp mask applied
		img.read(ifilename, data_in);

		data_sharp.resize(img.w * img.h * img.nchannels);

 		std::cout << "\n Running Tests...\n";
		std::cout << "\n-------------------------\n";

		for (int noOfTests = 0; noOfTests <= 6; noOfTests++){
			//start timer
		auto t1 = std::chrono::steady_clock::now();
		//code using the kernel
		unsharp_mask_cl(data_sharp.data(), data_in.data(), blur_radius, img.w, img.h, img.nchannels);
		//Serial code
		//unsharp_mask(data_sharp.data(), data_in.data(), blur_radius, img.w, img.h, img.nchannels);

		auto t2 = std::chrono::steady_clock::now(); // end timer
		std::cout << "Test " << noOfTests << " = " << std::chrono::duration<double>(t2 - t1).count() << " seconds.\n" << std::endl; // output test durations
		}
		std::cout << "\n-------------------------\n";
		std::cout << "\nOutputting Image...\n";
		std::cout << "\n-------------------------\n";

		// write image to out.ppm
		img.write(ofilename, data_sharp);

		std::cout << "\nImage output complete\n";
		std::cout << "\n-------------------------\n";
		return 0;
	}