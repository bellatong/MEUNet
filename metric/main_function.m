algorithms = {
     %'BASNet';
     %'BMPM';
     %'C2SNet';
     %'CPD';
    % 'DSS';
    % 'EGNet';

     %'F3Net';
     %'MLMSNet';
     %'MWS';
    % 'PAGE';
     %'PICANet';
     %'R3Net';
     %'RAS';
     %'TDBU';
     %'U2Net';
     %'u3net-wo-edbedge'
     'U3Net'
    };

datasets = {
              'DUTS-TE';
              'DUT-OMROM';
              'ECSSD';
              'PASCAL-S';
              'HKU-IS';
              'SOD';
              
    };


for i = 1:numel(algorithms)
    alg = algorithms{i};
    fprintf('The method is %s\n', alg);
    
%     fid = fopen(['/home/bianyetong/saliency_models/U3Net_master/save_datas/U3Net_epoch_50_itr_33660_train_0.288188/result.txt'],'a');
    fid = fopen(['/home/bianyetong/saliency_models/U3Net/test_data/U3Net_epoch_54_itr_36000_train_0.182058_lr_0.00000100_results/result.txt'],'a')
    fprintf(fid,'		maxF	fm	MAE	wfm	S-m	em\n');
    fclose(fid);
    
    for j = 1:numel(datasets)
        dataset      = datasets{j}
%         modelpath = '/home/bianyetong/saliency_models/'
%         testdata = '/save_datas/'
%         modelname = 'U3Net_epoch_50_itr_33660_train_0.288188/'
% 
        modelpath = '/home/bianyetong/saliency_models/'
        testdata = '/test_data/'
        modelname = 'U3Net_epoch_54_itr_36000_train_0.182058_lr_0.00000100_results/'
      
        predpath = strcat(modelpath,alg,testdata,modelname,dataset,'/')

        if ~exist(predpath, 'dir'), continue; end

        if strcmp(dataset, 'DUTS-TE')
            %predpath     = ['/home/bianyetong/saliency_models/U3Net/test_data/U3Net_bce_itr_66000_train_6.851483_results/DUTS/'];
            maskpath     = ['/home/bianyetong/datasets/DUTS/DUTS-TE/Mask/'];
            names = dir(['/home/bianyetong/datasets/DUTS/DUTS-TE/Mask/*.png']);
        elseif strcmp(dataset,'DUT-OMROM')
            %predpath     = ['/home/bianyetong/saliency_models/U3Net/test_data/U3Net_bce_itr_66000_train_6.851483_results/DUT-OMRON/'];
            maskpath     = ['/home/bianyetong/datasets/DUT-OMROM/Mask/'];            
            names = dir(['/home/bianyetong/datasets/DUT-OMROM/Mask/*.png']);
        elseif strcmp(dataset,'ECSSD')
        	%predpath     = ['/home/bianyetong/saliency_models/U3Net/test_data/U3Net_bce_itr_66000_train_6.851483_results/ECSSD/'];
        	maskpath     = ['/home/bianyetong/datasets/ECSSD/Mask/'];
            names = dir(['/home/bianyetong/datasets/ECSSD/Mask/*.png']);
        elseif strcmp(dataset,'PASCAL-S')
        	%predpath     = ['/home/bianyetong/saliency_models/U3Net/test_data/U3Net_bce_itr_66000_train_6.851483_results/PASCAL-S/'];
        	maskpath     = ['/home/bianyetong/datasets/PASCAL-S/Imgs/']; 
            names = dir(['/home/bianyetong/datasets/PASCAL-S/Imgs/*.png']);
        elseif strcmp(dataset,'HKU-IS')
            %predpath     = ['/home/bianyetong/saliency_models/U3Net/test_data/U3Net_bce_itr_66000_train_6.851483_results/HKU-IS/'];
            maskpath     = ['/home/bianyetong/datasets/HKU-IS/GT/']; 
            names = dir(['/home/bianyetong/datasets/HKU-IS/GT/*.png']);
        %elseif strcmp(dataset,'SOD')
        else
        	%predpath     = ['/home/bianyetong/saliency_models/U3Net/test_data/U3Net_bce_itr_66000_train_6.851483_results/SOD/'];
        	maskpath     = ['/home/bianyetong/datasets/SOD/Imgs/Imgs/']; 
            names = dir(['/home/bianyetong/datasets/SOD/Imgs/Imgs/*.png']);
        end


        

%{
        if strcmp(dataset, 'DUT-TE')
        	names = dir(['/home/bianyetong/datasets/DUTS/DUTS-TE/Mask/*.png']);
        elseif strcmp(dataset,'DUT-OMRON')
            names = dir(['/home/bianyetong/datasets/DUT-OMROM/Mask/*.png']);
        elseif strcmp(dataset,'ECSSD')
        	names = dir(['/home/bianyetong/datasets/ECSSD/Mask/*.png']);
        elseif strcmp(dataset,'PASCAL-S')
        	names = dir(['/home/bianyetong/datasets/PASCAL-S/Imgs/*.png']);
        elseif strcmp(dataset,'HKU-IS')
        	names = dir(['/home/bianyetong/datasets/HKU-IS/GT/*.png']);
        else
        	names = dir(['/home/bianyetong/datasets/SOD/Imgs/Imgs/*.png']);
        end
%}
        names = {names.name}';
        wfm          = 0; mae    = 0; sm     = 0; fm     = 0; prec   = 0; rec    = 0; em     = 0;
        score1       = 0; score2 = 0; score3 = 0; score4 = 0; score5 = 0; score6 = 0; score7 = 0;

        results      = cell(numel(names), 6);
        ALLPRECISION = zeros(numel(names), 256);
        ALLRECALL    = zeros(numel(names), 256);
        file_num     = false(numel(names), 1);
        for k = 1:numel(names)
            name          = names{k,1}
            results{k, 1} = name;
            file_num(k)   = true;
            fgpath        = [predpath name];
            fg            = imread(fgpath);
            
            if strcmp(alg, 'PoolNet')
                gtpath = [maskpath strrep(name, '_sal_fuse.', '.')];
            elseif strcmp(alg, 'CNet')
                gtpath = [maskpath strrep(name, '_10k.', '.')];
            else
                %gtpath = [maskpath name];
                gtpath = [maskpath name(1:end-4) '.png'];
            end
            gt = imread(gtpath);

            if length(size(fg)) == 3, fg = fg(:,:,1); end
            if length(size(gt)) == 3, gt = gt(:,:,1); end
            fg = imresize(fg, size(gt)); 
            fg = mat2gray(fg); 
            gt = mat2gray(gt);
            if max(fg(:)) == 0 || max(gt(:)) == 0, continue; end
            
            gt(gt>=0.5) = 1; gt(gt<0.5) = 0; gt = logical(gt);
            score1                   = MAE(fg, gt);
            [score2, score3, score4] = Fmeasure(fg, gt, size(gt)); 
            score5                   = wFmeasure(fg, gt); 
            score6                   = Smeasure(fg, gt);
            score7                   = Emeasure(fg, gt);
            mae                      = mae  + score1;
            prec                     = prec + score2;
            rec                      = rec  + score3;
            fm                       = fm   + score4;
            wfm                      = wfm  + score5;
            sm                       = sm   + score6;
            em                       = em   + score7;
            results{k, 2}            = score1; 
            results{k, 3}            = score4; 
            results{k, 4}            = score5; 
            results{k, 5}            = score6;
            results{k, 6}            = score7;
            [precision, recall]      = PRCurve(fg*255, gt);
            ALLPRECISION(k, :)       = precision;
            ALLRECALL(k, :)          = recall;
        end
        prec     = mean(ALLPRECISION(file_num,:), 1);   
        rec      = mean(ALLRECALL(file_num,:), 1);
        maxF     = max(1.3*prec.*rec./(0.3*prec+rec+eps));
        file_num = double(file_num);
        fm       = fm  / sum(file_num);
        mae      = mae / sum(file_num); 
        wfm      = wfm / sum(file_num); 
        sm       = sm  / sum(file_num); 
        em       = em  / sum(file_num);
%         fid = fopen(['/home/bianyetong/saliency_models/U3Net_master/save_datas/U3Net_epoch_50_itr_33660_train_0.288188/result.txt'],'a');
        fid = fopen(['/home/bianyetong/saliency_models/U3Net/test_data/U3Net_epoch_54_itr_36000_train_0.182058_lr_0.00000100_results/result.txt'],'a');

        fprintf(fid,'%s:    %6.3f, %6.3f, %6.3f, %6.3f, %6.3f, %6.3f\n',dataset, maxF, fm, mae, wfm, sm, em);
        fclose(fid);
%        fprintf('%s: %6.3f, %6.3f, %6.3f, %6.3f, %6.3f, %6.3f\n', dataset, maxF, fm, mae, wfm, sm, em);
        %save_path = ['./P-R' filesep alg filesep dataset filesep];
        %if ~exist(save_path, 'dir'), mkdir(save_path); end
        %save([save_path 'results.mat'], 'results');
        %save([save_path 'prec.mat'], 'prec');
        %save([save_path 'rec.mat'], 'rec');
    end
end
