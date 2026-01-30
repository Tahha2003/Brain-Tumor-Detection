function brain_mri_gui_original

    % Create GUI window
    fig = uifigure('Name','Brain MRI Tumor Detection',...
        'Position',[100 100 1100 650]);

    % Load Image Button
    uibutton(fig,'Text','Load MRI Image',...
        'Position',[20 600 150 30],...
        'ButtonPushedFcn',@(btn,event) loadImage());

    % Axes
    ax1 = uiaxes(fig,'Position',[50 330 300 250]);   % Original
    ax2 = uiaxes(fig,'Position',[400 330 300 250]);  % Binarized
    ax3 = uiaxes(fig,'Position',[750 330 300 250]);  % Watershed
    ax4 = uiaxes(fig,'Position',[200 40 300 250]);   % Morphology tumor
    ax5 = uiaxes(fig,'Position',[600 40 300 250]);   % Thresholding result

    % Text Output
    txt = uitextarea(fig,'Position',[20 40 150 200],...
        'Editable','off');

    function loadImage()

        [file,path] = uigetfile({'*.png;*.jpg;*.jpeg'});
        if isequal(file,0)
            return;
        end

        % Read image
        mri = imread(fullfile(path,file));
        mri = imresize(mri,[200 200]);
        mri_gray = im2gray(mri);

        imshow(mri_gray,'Parent',ax1);
        title(ax1,'Grayscale MRI');

        % -----------------------
        % Binary Image
        % -----------------------
        mri_bin = imbinarize(mri_gray,0.6);
        imshow(mri_bin,'Parent',ax2);
        title(ax2,'Binary Image');

        % -----------------------
        % Sobel + Watershed
        % -----------------------
        fx = fspecial('sobel');
        fy = fx';
        gx = imfilter(double(mri_bin),fx,'replicate');
        gy = imfilter(double(mri_bin),fy,'replicate');
        gradmag = sqrt(gx.^2 + gy.^2);

        ws = watershed(gradmag);
        ws_rgb = label2rgb(ws);
        imshow(ws_rgb,'Parent',ax3);
        title(ax3,'Watershed Segmentation');

        % -----------------------
        % Morphological Processing
        % -----------------------
        se = strel('disk',5);
        openImg = imopen(mri_bin,se);
        recon = imreconstruct(openImg,mri_bin);
        dil = imdilate(recon,se);
        comp = imreconstruct(imcomplement(dil),imcomplement(recon));
        tumor = imcomplement(comp);

        imshow(tumor,'Parent',ax4);
        title(ax4,'Morphological Tumor');

        % -----------------------
        % 🔹 THRESHOLDING BASED SEGMENTATION 🔹
        % -----------------------
        level = graythresh(mri_gray);   % Otsu threshold
        threshImg = imbinarize(mri_gray,level);
        threshImg = bwareaopen(threshImg,50);

        imshow(threshImg,'Parent',ax5);
        title(ax5,'Thresholding Segmentation');

        % -----------------------
        % Tumor Size Calculation
        % -----------------------
        cc = bwconncomp(tumor);
        props = regionprops(cc,'Area');

        pixel_w = 0.0508;
        pixel_h = 0.0508;

        tumor_pixels = sum([props.Area]);
        tumor_cm2 = tumor_pixels * pixel_w * pixel_h;

        if tumor_cm2 == 0
            category = "No Tumor";
        elseif tumor_cm2 <= 2.37
            category = "Benign Tumor";
        else
            category = "Malignant Tumor";
        end

        % Display Text Output
        txt.Value = {
            'Tumor Analysis'
            '-------------------'
            ['Area (cm^2): ' num2str(tumor_cm2)]
            ['Category: ' char(category)]
            };

    end
end

        % -----------------------
        % 🔹Code is Written by Rizwan Aleem Tahha | M Ahsan | Moiz Ahmed | Saim Asad 🔹
        % -----------------------