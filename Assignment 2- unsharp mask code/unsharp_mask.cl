
// GPGPU Assignment 2
// Unsharp Mask
// B00268411 Paul McLaughlin
inline void pixel_average(__global uchar *out,
	__global uchar *in,
	const int x, const int y, const int blur_radius,
	const unsigned width, const unsigned height, const unsigned nchannels)
{
	float red_total = 0, green_total = 0, blue_total = 0;

	for (int j = y - blur_radius + 1; j < y + blur_radius; ++j) 
	{
		for (int i = x - blur_radius + 1; i < x + blur_radius; ++i) 
		{
			const unsigned r_i = i < 0 ? 0 : i >= width ? width - 1 : i;
			const unsigned r_j = j < 0 ? 0 : j >= height ? height - 1 : j;
			unsigned byte_offset = (r_j*width + r_i)*nchannels;
			red_total += in[byte_offset + 0];
			green_total += in[byte_offset + 1];
			blue_total += in[byte_offset + 2];
		}
	}

	const unsigned nsamples = (blur_radius * 2 - 1) * (blur_radius * 2 - 1);
	unsigned byte_offset = (y*width + x)*nchannels;
	out[byte_offset + 0] = red_total / nsamples;
	out[byte_offset + 1] = green_total / nsamples;
	out[byte_offset + 2] = blue_total / nsamples;
}


__kernel void blur(__global uchar *out, __global uchar *in,
	const int blur_radius,
	const int width, const int height, const unsigned nchannels)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	pixel_average(out, in, x, y, blur_radius, width, height, nchannels);
}


__kernel void add_weighted(__global uchar *out, __global uchar *in, __global uchar *blur, const unsigned width, const unsigned height, const float alpha, const float beta, const float gamma,
	 const unsigned nchannels)
{
	int y = get_global_id(1);
	int x = get_global_id(0);
	unsigned byte_offset = (y*width + x)*nchannels;

	float f = in[byte_offset + 0] * alpha + blur[byte_offset + 0] * beta + gamma;
	out[byte_offset + 0] = f < 0 ? 0 : f > UCHAR_MAX ? UCHAR_MAX : f;

	f = in[byte_offset + 1] * alpha + blur[byte_offset + 1] * beta + gamma;
	out[byte_offset + 1] = f < 0 ? 0 : f > UCHAR_MAX ? UCHAR_MAX : f;

	f = in[byte_offset + 2] * alpha + blur[byte_offset + 2] * beta + gamma;
	out[byte_offset + 2] = f < 0 ? 0 : f > UCHAR_MAX ? UCHAR_MAX : f;
}