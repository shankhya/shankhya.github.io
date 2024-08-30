#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <lcms2.h>

// Function prototypes
cmsHPROFILE make_sRGB_profile(cmsCIExyY whitepoint, cmsCIExyYTRIPLE primaries, char* trc, char* basename, char* id, char* extension, cmsMLU* copyright, char* manufacturer);
cmsToneCurve* make_tonecurve(char* trc);
char* make_file_name(char* basename, char* id, char* profile_version, char* trc, char* extension);

int main() {
    cmsHPROFILE sRGB_profile;
    cmsCIExyY whitepoint;
    cmsCIExyYTRIPLE primaries;
    cmsMLU* copyright = cmsMLUalloc(NULL, 1);
    cmsMLUsetASCII(copyright, "en", "US", "Shankhya Debnath, CC-BY-SA 3.0");
    char* basename = "sRGB";
    char* trc = "-srgbtrc";
    char* manufacturer = "sRGB custom profile generator";
    char* id = "-custom", *extension = ".icc";

    // Define white points and primaries
    cmsCIExyY d65_whitepoint = {0.3127, 0.3290, 1.0};
    cmsCIExyYTRIPLE srgb_primaries = {
        {0.639998686, 0.330010138, 1.0},
        {0.300003784, 0.600003357, 1.0},
        {0.150002046, 0.059997204, 1.0}
    };

    // Set the whitepoint and primaries
    whitepoint = d65_whitepoint;
    primaries = srgb_primaries;

    // Create sRGB profile
    sRGB_profile = make_sRGB_profile(whitepoint, primaries, trc, basename, id, extension, copyright, manufacturer);

    // Clean up
    cmsMLUfree(copyright);
    return 0;
}

cmsHPROFILE make_sRGB_profile(cmsCIExyY whitepoint, cmsCIExyYTRIPLE primaries, char* trc, char* basename, char* id, char* extension, cmsMLU* copyright, char* manufacturer) {
    cmsToneCurve* curve[3], * tonecurve;
    tonecurve = make_tonecurve(trc);
    curve[0] = curve[1] = curve[2] = tonecurve;

    // Create sRGB profile
    cmsHPROFILE sRGB_profile = cmsCreateRGBProfile(&whitepoint, &primaries, curve);
    cmsWriteTag(sRGB_profile, cmsSigCopyrightTag, copyright);

    cmsMLU* MfgDesc = cmsMLUalloc(NULL, 1);
    cmsMLUsetASCII(MfgDesc, "en", "US", manufacturer);
    cmsWriteTag(sRGB_profile, cmsSigDeviceMfgDescTag, MfgDesc);

    char* profile_version = "-V4";
    char* filename = make_file_name(basename, id, profile_version, trc, extension);
    char* description_text = filename + 12;
    cmsMLU* description = cmsMLUalloc(NULL, 1);
    cmsMLUsetASCII(description, "en", "US", description_text);
    cmsWriteTag(sRGB_profile, cmsSigProfileDescriptionTag, description);
    cmsSaveProfileToFile(sRGB_profile, filename);
    cmsMLUfree(description);
    cmsMLUfree(MfgDesc);

    return sRGB_profile;
}

cmsToneCurve* make_tonecurve(char* trc) {
    cmsToneCurve* tonecurve;
    if (strcmp(trc, "-srgbtrc") == 0) {
        cmsFloat64Number srgb_parameters[5] = { 2.4, 1.0 / 1.055, 0.055 / 1.055, 1.0 / 12.92, 0.04045 };
        tonecurve = cmsBuildParametricToneCurve(NULL, 4, srgb_parameters);
    }
    return tonecurve;
}

char* make_file_name(char* basename, char* id, char* profile_version, char* trc, char* extension) {
    int len = strlen(basename) + strlen(id) + strlen(profile_version) + strlen(trc) + strlen(extension) + 1;
    char* filename = (char*)malloc(len);
    if (filename == NULL) {
        printf("Error allocating memory for filename\\n");
        exit(1);
    }
    strcpy(filename, basename);
    strcat(filename, id);
    strcat(filename, profile_version);
    strcat(filename, trc);
    strcat(filename, extension);
    return filename;
}
