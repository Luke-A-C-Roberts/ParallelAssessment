//a simple OpenCL kernel which copies all pixels from A to B
kernel void identity(global const uchar* A, global uchar* B) {
	int id   = get_global_id(0);
	int size = get_global_size(0);

	if (id % (size / 3) == 0)
		printf("%d: %d\n", id, A[id]);

	B[id] = A[id];
}

kernel void u8hsl(global const uchar* input, global uchar* output) {
	int size = get_global_size(0);
	int gid = get_global_id(0);

	if (gid > size / 3)
		return;

	float r = ((float)input[gid]) / 255.0;
	float g = ((float)input[gid + size / 3]) / 255.0;
	float b = ((float)input[gid + 2 * size / 3]) / 255.0;

	// https://www.niwa.nu/2013/05/math-behind-colorspace-conversions-rgb-hsl/
	float cmax = fmax(fmax(r, g), b);
	float cmin = fmin(fmin(r, g), b);
	float cdelta = cmax - cmin;
	float csum = cmax + cmin;

	float l = csum / 2.0;
	float s = (l <= 0.5) ? (cdelta / csum) : (cdelta / (2.0 - cmax - cmin));
	float h;

	if (cmax == r) {
		h = (g - b) / cdelta;
	}
	else if (cmax == g) {
		h = 2.0 + (b - r) / cdelta;
	}
	else if (cmax == b) {
		h = 4.0 + (r - g) / cdelta;
	}

	output[gid] = (uchar)(h * 255);
	output[gid + size / 3] = (uchar)(s * 255);
	output[gid + 2 * size / 3] = (uchar)(l * 255);
}



//kernel void u8histogram(global const uchar* input, global uint* output, local uint* sums, uchar*) {
//	int size = get_global_size(0);
//	int gid = get_global_id(0);
//
//	if (gid > size / 3)
//		return;
//
//	
//}