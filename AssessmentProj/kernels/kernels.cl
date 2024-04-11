float calculate_cmyk_band(float color, float k) {
	return (1. - color - k) / (1. - k);	
}

float insure_cmyk_range(float color) {
	return fmax(0, fmin(1, color));
}

kernel void uchar_rgb_to_cmyk(global const uchar* in, global uchar* out) {
	int size = get_global_size(0);
	int gid = get_global_id(0);
	int channel_size = size / 3;
	
	if (gid > size / 3)
		return;
	
	float r = ((float)in[gid]) / 255.;
	float g = ((float)in[gid + channel_size]) / 255.;
	float b = ((float)in[gid + channel_size * 2]) / 255.;
	
	float k = 1. - fmax(r, fmax(g, b));
	float c = insure_cmyk_range(calculate_cmyk_band(r, k));
	float m = insure_cmyk_range(calculate_cmyk_band(g, k));
	float y = insure_cmyk_range(calculate_cmyk_band(b, k));
	k = insure_cmyk_range(k);

	out[gid] = (uchar)(c * 255.);
	out[gid + channel_size] = (uchar)(m * 255.);
	out[gid + channel_size * 2] = (uchar)(y * 255.);
	out[gid + channel_size * 3] = (uchar)(k * 255.);
}

kernel void ushort_rgb_to_cmyk(global const ushort* in, global ushort* out) {
	int size = get_global_size(0);
	int gid = get_global_id(0);
	int channel_size = size / 3;
	
	if (gid > size / 3)
		return;
	
	float r = ((float)in[gid]) / 255.;
	float g = ((float)in[gid + channel_size]) / 255.;
	float b = ((float)in[gid + channel_size * 2]) / 255.;
	
	float k = 1. - fmax(r, fmax(g, b));
	float c = insure_cmyk_range(calculate_cmyk_band(r, k));
	float m = insure_cmyk_range(calculate_cmyk_band(g, k));
	float y = insure_cmyk_range(calculate_cmyk_band(b, k));
	k = insure_cmyk_range(k);

	out[gid] = (ushort)(c * 255.);
	out[gid + channel_size] = (ushort)(m * 255.);
	out[gid + channel_size * 2] = (ushort)(y * 255.);
	out[gid + channel_size * 3] = (ushort)(k * 255.);
}

kernel void uchar_hist(global const uchar* in, global uint* hist, local uint* local_hist, const ulong bins) {
	int gid = get_global_id(0);
	int gsize = get_global_size(0);
	int lid = get_local_id(0);
	int lsize = get_global_size(0);
	
	if (lid < bins)
		local_hist[lid] = 0;

	barrier(CLK_LOCAL_MEM_FENCE);
	
	if (gid < gsize)
		atomic_inc(&local_hist[in[gid]]);

	barrier(CLK_LOCAL_MEM_FENCE);

	if (lid < bins)
		atomic_add(&hist[lid], local_hist[lid]);
}

kernel void ushort_hist(global const ushort* in, global uint* hist, local uint* local_hist, const ulong bins) {
	int gid = get_global_id(0);
	int gsize = get_global_size(0);

	if (gid < gsize) {
		atomic_inc(&hist[in[gid]]);
	}

}

int blelloch_r(const int gid, const int stride) {
	return (gid + 1) % (stride * 2);
}

kernel void uchar_cdf(global uint* in, global uchar* out, const ulong pixels, ulong bins) {
	int gid = get_global_id(0);
	int gsize = get_global_size(0);
	int temp, relative_gid;

	for (int stride = 1; stride < gsize; stride *= 2) {
		if (!blelloch_r(gid, stride)) in[gid] += in[gid - stride];
		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	if (!gid) in[gsize-1] = 0;
	barrier(CLK_GLOBAL_MEM_FENCE);
	for (int stride = gsize/2; stride > 0; stride /= 2) {
		if (!blelloch_r(gid, stride)) {
			relative_gid = gid - stride;
			temp = in[gid];
			in[gid] += in[relative_gid];
			in[relative_gid] = temp;
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	bins--;
	out[gid] = (uchar)round(
		((float)in[gid] - (float)in[0]) * bins
		/ ((float)in[bins] - (float)in[0])
	);
}

kernel void ushort_cdf(global uint* in, global ushort* out, const ulong pixels, ulong bins) {
	int gid = get_global_id(0);
	int gsize = get_global_size(0);
	int temp, relative_gid;

	for (int stride = 1; stride < gsize; stride *= 2) {
		if (!blelloch_r(gid, stride)) in[gid] += in[gid - stride];
		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	if (!gid) in[gsize-1] = 0;
	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int stride = gsize/2; stride > 0; stride /= 2) {
		if (!blelloch_r(gid, stride)) {
			relative_gid = gid - stride;
			temp = in[gid];
			in[gid] += in[relative_gid];
			in[relative_gid] = temp;
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	bins--;
	out[gid] = (ushort)round(
		((float)in[gid] - (float)in[0]) * bins
		/ ((float)in[bins] - (float)in[0])
	);
}

kernel void uchar_cdf_lookup(global uchar* light_vals, global const uchar* cdf) {
	int gid = get_global_id(0);
	int gsize = get_global_size(0);

	light_vals[gid] = cdf[light_vals[gid]];
}

kernel void ushort_cdf_lookup(global ushort* light_vals, global const ushort* cdf) {
	int gid = get_global_id(0);
	int gsize = get_global_size(0);

	light_vals[gid] = cdf[light_vals[gid]];
}

float calculate_rgb_band(float color, float k) {
	return (1. - color) * (1. - k);
}

float insure_rgb_range(float color) {
	return fmax(0, fmin(1, color));
}

kernel void uchar_cmyk_to_rgb(global const uchar* in, global uchar* out) {
	int size = get_global_size(0);
	int gid = get_global_id(0);
	int channel_size = size / 4;
	
	if (gid > size / 4)
		return;

	float c = ((float)in[gid]) / 255.;
	float m = ((float)in[gid + channel_size]) / 255.;
	float y = ((float)in[gid + channel_size * 2]) / 255.;
	float k = ((float)in[gid + channel_size * 3]) / 255.;
	
	float r = insure_rgb_range(calculate_rgb_band(c, k));
	float g = insure_rgb_range(calculate_rgb_band(m, k));
	float b = insure_rgb_range(calculate_rgb_band(y, k));

	out[gid] = (uchar)(r * 255);
	out[gid + channel_size] = (uchar)(g * 255);
	out[gid + channel_size * 2] = (uchar)(b * 255);	
}

kernel void ushort_cmyk_to_rgb(global const ushort* in, global ushort* out) {
	int size = get_global_size(0);
	int gid = get_global_id(0);
	int channel_size = size / 4;
	
	if (gid > size / 4)
		return;

	float c = ((float)in[gid]) / 255.;
	float m = ((float)in[gid + channel_size]) / 255.;
	float y = ((float)in[gid + channel_size * 2]) / 255.;
	float k = ((float)in[gid + channel_size * 3]) / 255.;
	
	float r = insure_rgb_range(calculate_rgb_band(c, k));
	float g = insure_rgb_range(calculate_rgb_band(m, k));
	float b = insure_rgb_range(calculate_rgb_band(y, k));

	out[gid] = (ushort)(r * 255);
	out[gid + channel_size] = (ushort)(g * 255);
	out[gid + channel_size * 2] = (ushort)(b * 255);	
}
