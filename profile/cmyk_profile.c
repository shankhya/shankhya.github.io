#include <lcms2.h>
#include <stdio.h>
#include <stdlib.h>

#define CLUT_SIZE 3  // Reduced size for testing

// Function to create a complete CMYK printer profile
cmsHPROFILE create_complete_cmyk_profile() {
    cmsHPROFILE hProfile = NULL;
    cmsCIEXYZ D50_XYZ = {0.9642, 1.0000, 0.8249}; // D50 white point
    cmsCIEXYZ D65_XYZ = {0.95047, 1.00000, 1.08883}; // D65 white point (commonly used as an illuminant)
    cmsCIEXYZ surround = {0.3457, 0.3585, 0.0}; // Typical surround condition
    cmsCIEXYZ illuminant = D65_XYZ; // Illuminant (using D65)
    cmsCIExyY D50_xyY;
    cmsToneCurve* CMYKcurves[4] = {NULL, NULL, NULL, NULL};
    cmsPipeline *AToB0 = NULL, *BToA0 = NULL;
    cmsStage* clutStageA2B0 = NULL, *clutStageBToA0 = NULL;
    cmsPipeline* gamutPipeline = NULL;

    cmsUInt16Number* A2BTable = NULL;
    cmsUInt16Number* B2ATable = NULL;

    printf("Starting profile creation...\n");

    // Convert D50_XYZ to D50_xyY
    cmsXYZ2xyY(&D50_xyY, &D50_XYZ);
    printf("Converted D50_XYZ to D50_xyY.\n");

    // Initialize tone curves (identity curves)
    for (int i = 0; i < 4; i++) {
        CMYKcurves[i] = cmsBuildGamma(NULL, 1.0);
        if (!CMYKcurves[i]) {
            fprintf(stderr, "Error creating tone curve for channel %d\n", i);
            goto cleanup;
        }
        printf("Created tone curve for channel %d.\n", i);
    }

    // Create an empty CMYK profile
    hProfile = cmsCreateProfilePlaceholder(NULL);
    if (!hProfile) {
        fprintf(stderr, "Error creating profile placeholder\n");
        goto cleanup;
    }
    printf("Created profile placeholder.\n");

    // Set the manufacturer name in the profile header (using a custom signature)
    cmsSetHeaderManufacturer(hProfile, (cmsTagSignature) ('T' | ('P' << 8) | ('I' << 16) | ('R' << 24))); // Example: 'MFGR' for Manufacturer

    // Set the device class in the profile header
    cmsSetDeviceClass(hProfile, cmsSigOutputClass); // Setting the device class to "Output"

    // Set the profile to be CMYK
    cmsSetPCS(hProfile, cmsSigLabData);
    cmsSetColorSpace(hProfile, cmsSigCmykData);
    printf("Set PCS and color space.\n");

    // Set the white point
    if (!cmsWriteTag(hProfile, cmsSigMediaWhitePointTag, &D50_XYZ)) {
        fprintf(stderr, "Error writing white point tag\n");
        goto cleanup;
    }
    printf("Set the white point.\n");

    // Add the copyright (cprt) tag
    cmsMLU* copyright = cmsMLUalloc(NULL, 1);
    if (!copyright) {
        fprintf(stderr, "Error allocating memory for copyright MLU\n");
        goto cleanup;
    }
    cmsMLUsetASCII(copyright, "en", "US", "Shankhya Debnath");
    if (!cmsWriteTag(hProfile, cmsSigCopyrightTag, copyright)) {
        fprintf(stderr, "Error writing copyright tag\n");
        cmsMLUfree(copyright);
        goto cleanup;
    }
    cmsMLUfree(copyright);
    printf("Added copyright (cprt) tag.\n");

    // Add the description (desc) tag
    cmsMLU* description = cmsMLUalloc(NULL, 1);
    if (!description) {
        fprintf(stderr, "Error allocating memory for description MLU\n");
        goto cleanup;
    }
    cmsMLUsetASCII(description, "en", "US", "A simple CMYK printer profile");
    if (!cmsWriteTag(hProfile, cmsSigProfileDescriptionTag, description)) {
        fprintf(stderr, "Error writing description tag\n");
        cmsMLUfree(description);
        goto cleanup;
    }
    cmsMLUfree(description);
    printf("Added description (desc) tag.\n");

    // Allocate memory for CLUT tables
    size_t A2BTableSize = CLUT_SIZE * CLUT_SIZE * CLUT_SIZE * 3 * sizeof(cmsUInt16Number);
    size_t B2ATableSize = CLUT_SIZE * CLUT_SIZE * CLUT_SIZE * 4 * sizeof(cmsUInt16Number);
    A2BTable = (cmsUInt16Number*)malloc(A2BTableSize);
    B2ATable = (cmsUInt16Number*)malloc(B2ATableSize);
    if (!A2BTable || !B2ATable) {
        fprintf(stderr, "Error allocating memory for CLUT tables\n");
        goto cleanup;
    }
    printf("Allocated memory for CLUT tables.\n");

    // Initialize A2BTable and B2ATable with simple identity transformation
    for (size_t i = 0; i < A2BTableSize / sizeof(cmsUInt16Number); i++) {
        A2BTable[i] = (cmsUInt16Number)(i % 65536);  // Simple initialization
    }
    for (size_t i = 0; i < B2ATableSize / sizeof(cmsUInt16Number); i++) {
        B2ATable[i] = (cmsUInt16Number)(i % 65536);  // Simple initialization
    }
    printf("Initialized CLUT tables.\n");

    // Create A2B0 LUT: CMYK -> Lab
    AToB0 = cmsPipelineAlloc(NULL, 4, 3);
    if (!AToB0) {
        fprintf(stderr, "Error allocating AToB0 pipeline\n");
        goto cleanup;
    }
    printf("Allocated AToB0 pipeline.\n");

    // Allocate and insert CLUT stage into AToB0 pipeline
    clutStageA2B0 = cmsStageAllocCLut16bit(NULL, CLUT_SIZE, 4, 3, A2BTable);
    if (!clutStageA2B0) {
        fprintf(stderr, "Error allocating CLUT stage for AToB0\n");
        cmsPipelineFree(AToB0);
        goto cleanup;
    }
    if (!cmsPipelineInsertStage(AToB0, cmsAT_END, clutStageA2B0)) {
        fprintf(stderr, "Error inserting CLUT into AToB0 pipeline\n");
        cmsStageFree(clutStageA2B0);
        cmsPipelineFree(AToB0);
        goto cleanup;
    }
    if (!cmsWriteTag(hProfile, cmsSigAToB0Tag, AToB0)) {
        fprintf(stderr, "Error writing AToB0 tag\n");
        cmsStageFree(clutStageA2B0);
        cmsPipelineFree(AToB0);
        goto cleanup;
    }
    printf("Created and inserted AToB0 LUT.\n");

    // Create B2A0 LUT: Lab -> CMYK
    BToA0 = cmsPipelineAlloc(NULL, 3, 4);
    if (!BToA0) {
        fprintf(stderr, "Error allocating BToA0 pipeline\n");
        goto cleanup;
    }
    printf("Allocated BToA0 pipeline.\n");

    // Allocate and insert CLUT stage into BToA0 pipeline
    clutStageBToA0 = cmsStageAllocCLut16bit(NULL, CLUT_SIZE, 3, 4, B2ATable);
    if (!clutStageBToA0) {
        fprintf(stderr, "Error allocating CLUT stage for BToA0\n");
        cmsPipelineFree(BToA0);
        goto cleanup;
    }
    if (!cmsPipelineInsertStage(BToA0, cmsAT_END, clutStageBToA0)) {
        fprintf(stderr, "Error inserting CLUT into BToA0 pipeline\n");
        cmsStageFree(clutStageBToA0);
        cmsPipelineFree(BToA0);
        goto cleanup;
    }
    if (!cmsWriteTag(hProfile, cmsSigBToA0Tag, BToA0)) {
        fprintf(stderr, "Error writing BToA0 tag\n");
        cmsStageFree(clutStageBToA0);
        cmsPipelineFree(BToA0);
        goto cleanup;
    }
    printf("Created and inserted BToA0 LUT.\n");

    // Create and write the gamut tag (gamt)
    gamutPipeline = cmsPipelineAlloc(NULL, 3, 3);  // 3 inputs (Lab), 3 outputs (Lab)
    if (!gamutPipeline) {
        fprintf(stderr, "Error allocating gamut pipeline\n");
        goto cleanup;
    }

    // Allocate and insert a CLUT stage for the gamut
    cmsStage* gamutCLUT = cmsStageAllocCLut16bit(NULL, CLUT_SIZE, 3, 3, NULL);  // NULL data initializes to identity
    if (!gamutCLUT) {
        fprintf(stderr, "Error allocating gamut CLUT stage\n");
        cmsPipelineFree(gamutPipeline);
        goto cleanup;
    }
    cmsPipelineInsertStage(gamutPipeline, cmsAT_END, gamutCLUT);

    if (!cmsWriteTag(hProfile, cmsSigGamutTag, gamutPipeline)) {
        fprintf(stderr, "Error writing gamut (gamt) tag\n");
        cmsPipelineFree(gamutPipeline);
        goto cleanup;
    }
    cmsPipelineFree(gamutPipeline);
    printf("Added gamut (gamt) tag.\n");

    // Add the vued (viewing conditions description) tag
    cmsMLU* viewingDesc = cmsMLUalloc(NULL, 1);
    if (!viewingDesc) {
        fprintf(stderr, "Error allocating memory for viewing conditions description MLU\n");
        goto cleanup;
    }
    cmsMLUsetASCII(viewingDesc, "en", "US", "D50");  // Example text
    if (!cmsWriteTag(hProfile, cmsSigViewingCondDescTag, viewingDesc)) {
        fprintf(stderr, "Error writing viewing conditions description (vued) tag\n");
        cmsMLUfree(viewingDesc);
        goto cleanup;
    }
    cmsMLUfree(viewingDesc);
    printf("Added viewing conditions description (vued) tag.\n");

    // Save the profile to disk
    if (!cmsSaveProfileToFile(hProfile, "Simple_CMYK_Profile.icc")) {
        fprintf(stderr, "Error saving profile\n");
        goto cleanup;
    }

    printf("Profile created and saved successfully.\n");

cleanup:
    if (hProfile) cmsCloseProfile(hProfile);
    if (AToB0) cmsPipelineFree(AToB0);
    if (BToA0) cmsPipelineFree(BToA0);
    if (clutStageA2B0) cmsStageFree(clutStageA2B0);
    if (clutStageBToA0) cmsStageFree(clutStageBToA0);
    for (int i = 0; i < 4; i++) {
        if (CMYKcurves[i]) cmsFreeToneCurve(CMYKcurves[i]);
    }

    return hProfile;
}

int main() {
    create_complete_cmyk_profile();
    return 0;
}
