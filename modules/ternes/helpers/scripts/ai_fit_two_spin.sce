clear();
clearglobal();
funcprot(0);

PATH=get_absolute_file_path("ai_fit_two_spin.sce");
ROOT=PATH+"../../";
exec(ROOT+"main/Spin-Hamiltonians-1.0.sce",-1);
exec(ROOT+"main/Spin-additional-functions-1.0.sce",-1);
exec(ROOT+"main/Spin-fit-functions-1.0.sce",-1);

global %savedata %plotspec %precission %normalize %zeroadj %stopcalc %weight;
global %Data %Data_trunc experiment experiment2 calc old_k itcount minvalues1 maxvalues1;

function v=env_or_empty(name)
    v="";
    ierr=execstr("v=getenv(name)","errcatch");
    if ierr<>0 then
        v="";
    end
endfunction

%savedata=%F;
%plotspec=%F;
%precission=0.001;
%normalize=%F;
%zeroadj=%T;
%stopcalc=%F;

minvalues1 = [0, -10, -10, -1, -5, 2, 0, -0.05, -5, -10, 0.1, -1, -20, -20, -20, 2];
maxvalues1 = [4, 10, 10, 1, 0, 200, 100, 0.05, 5, 10, 10, 1, 20, 20, 20, 200];

data_file=env_or_empty("AI_FIT_DATA");
if data_file=="" then
    data_file=ROOT+"io/input/origin_single_chain_1.txt";
end
output_file=env_or_empty("AI_FIT_OUTPUT");
if output_file=="" then
    output_file=ROOT+"io/output/ai_fit_two_spin_result.txt";
end

dat=fscanfMat(data_file);
if max(abs(dat(:,1)))<1 then
    %Data=[1000*dat(:,1), dat(:,2)];
else
    %Data=[dat(:,1), dat(:,2)];
end
x_scale=env_or_empty("AI_X_SCALE");
if x_scale<>"" then
    %Data(:,1)=%Data(:,1)*evstr(x_scale);
end
if max(abs(%Data(:,1)))>30 then
    %Data_trunc=%Data(find(abs(%Data(:,1))<=30),:);
else
    %Data_trunc=%Data;
end
%weight=ones(size(%Data_trunc,1),1);
if env_or_empty("AI_WEIGHT_NEGPEAK")=="1" then
    idx=find(%Data_trunc(:,1)>=-2.8 & %Data_trunc(:,1)<=-1.6);
    %weight(idx)=5;
    idx=find(%Data_trunc(:,1)>=-4.5 & %Data_trunc(:,1)<=-3.5);
    %weight(idx)=2;
end

atom1.S=0.5;
atom1.g=2;
atom1.D=0;
atom1.E=0;
atom1.J=-1.9653787;
atom1.U=-0.2060748;
atom1.w=7.7663185;

atom2.S=0.5;
atom2.g=2;
atom2.D=0;
atom2.E=0;
atom2.J=-0.04;
atom2.U=0;
atom2.w=20;

experiment.T=0.7045118;
experiment.xrange=30;
experiment.lt=0.005;
experiment.ptip=[0.0 0.0 0.0];
experiment.psample=[0 0 0];
experiment.atom=[atom1 atom2];
experiment.Eigenvec=[];
experiment.Eigenval=[];
experiment.position=1;
experiment.jposition=1;
experiment.A=12.991656;
experiment.b=0.0009958;
experiment.x0=0.6458583;
experiment.y0=3.216245;
experiment.B=[0 0 0];
experiment.matrix="1";
experiment.matrixDM="0, 0, 0";
experiment.heisenberg_coupling=2.3867879;
experiment.sample_entanglement=%T;
experiment.sef=1;
experiment.paramagnetic=%F;
experiment.paramag_S=5/2;
experiment.paramag_g=2;
experiment.eta=0.3;
experiment.no_eval=1000;
experiment.max_no_eigenstates=50;
experiment.third_order_calc=%T;
experiment.rate_calc=%F;
experiment.entanglement=%F;
experiment.allatomsequ=0;
init_c=env_or_empty("AI_INIT_C");
if init_c<>"" then
    experiment.heisenberg_coupling=evstr(init_c);
end
init_u1=env_or_empty("AI_INIT_U1");
if init_u1<>"" then
    experiment.atom(1).U=evstr(init_u1);
end
init_jrho1=env_or_empty("AI_INIT_JRHO1");
if init_jrho1<>"" then
    experiment.atom(1).J=evstr(init_jrho1);
end
init_w1=env_or_empty("AI_INIT_W1");
if init_w1<>"" then
    experiment.atom(1).w=evstr(init_w1);
end
init_x0=env_or_empty("AI_INIT_X0");
if init_x0<>"" then
    experiment.x0=evstr(init_x0);
end
init_t=env_or_empty("AI_INIT_TMEV");
if init_t<>"" then
    experiment.T=evstr(init_t);
end
init_lt=env_or_empty("AI_INIT_LT");
if init_lt<>"" then
    experiment.lt=evstr(init_lt);
end
init_a=env_or_empty("AI_INIT_A");
if init_a<>"" then
    experiment.A=evstr(init_a);
end
init_b=env_or_empty("AI_INIT_B");
if init_b<>"" then
    experiment.b=evstr(init_b);
end
init_y0=env_or_empty("AI_INIT_Y0");
if init_y0<>"" then
    experiment.y0=evstr(init_y0);
end

free_t=%T;
if env_or_empty("AI_FREE_T")=="0" then
    free_t=%F;
end
free_c=%T;
if env_or_empty("AI_FREE_C")=="0" then
    free_c=%F;
end

experiment.sw=[%F;%F;%F;%T;%T;%T;%T;%T;%T;%T;free_t;%F;free_c];

config_file=PATH+"ai_fit_config.sce";
if isfile(config_file) then
    exec(config_file,-1);
end
if env_or_empty("AI_NUM_SPINS")=="3" then
    atom1.S=0.5; atom1.g=2; atom1.D=0; atom1.E=0;
    atom1.U=0.05; atom1.J=-1.2; atom1.w=20;
    atom2.S=0.5; atom2.g=2; atom2.D=0; atom2.E=0;
    atom2.U=0.05; atom2.J=-1.2; atom2.w=20;
    atom3=atom1;
    experiment.atom=[atom1 atom2 atom3];
    experiment.matrix=["1","0";"0","1"];
    experiment.matrixDM=["0, 0, 0","0, 0, 0";"0, 0, 0","0, 0, 0"];
    experiment.heisenberg_coupling=1.4;
    experiment.sample_entanglement=%T;
    experiment.entanglement=%F;
end
init_c=env_or_empty("AI_INIT_C");
if init_c<>"" then
    experiment.heisenberg_coupling=evstr(init_c);
end
init_u1=env_or_empty("AI_INIT_U1");
if init_u1<>"" then
    experiment.atom(1).U=evstr(init_u1);
end
init_jrho1=env_or_empty("AI_INIT_JRHO1");
if init_jrho1<>"" then
    experiment.atom(1).J=evstr(init_jrho1);
end
init_w1=env_or_empty("AI_INIT_W1");
if init_w1<>"" then
    experiment.atom(1).w=evstr(init_w1);
end
init_x0=env_or_empty("AI_INIT_X0");
if init_x0<>"" then
    experiment.x0=evstr(init_x0);
end
init_t=env_or_empty("AI_INIT_TMEV");
if init_t<>"" then
    experiment.T=evstr(init_t);
end
init_a=env_or_empty("AI_INIT_A");
if init_a<>"" then
    experiment.A=evstr(init_a);
end
init_b=env_or_empty("AI_INIT_B");
if init_b<>"" then
    experiment.b=evstr(init_b);
end
init_y0=env_or_empty("AI_INIT_Y0");
if init_y0<>"" then
    experiment.y0=evstr(init_y0);
end
if env_or_empty("AI_FREE_X0")=="0" then
    experiment.sw(9)=%F;
end
if env_or_empty("AI_FREE_X0")=="1" then
    experiment.sw(9)=%T;
end
if env_or_empty("AI_FREE_C")=="0" then
    experiment.sw(13)=%F;
end
if env_or_empty("AI_FREE_C")=="1" then
    experiment.sw(13)=%T;
end
if env_or_empty("AI_FREE_T")=="0" then
    experiment.sw(11)=%F;
end
if env_or_empty("AI_FREE_T")=="1" then
    experiment.sw(11)=%T;
end
if env_or_empty("AI_FREE_B")=="0" then
    experiment.sw(8)=%F;
end
if env_or_empty("AI_FREE_B")=="1" then
    experiment.sw(8)=%T;
end
if env_or_empty("AI_FREE_U")=="0" then
    experiment.sw(4)=%F;
end
if env_or_empty("AI_FREE_U")=="1" then
    experiment.sw(4)=%T;
end
if env_or_empty("AI_FREE_JRHO")=="0" then
    experiment.sw(5)=%F;
end
if env_or_empty("AI_FREE_JRHO")=="1" then
    experiment.sw(5)=%T;
end
if env_or_empty("AI_FREE_W")=="0" then
    experiment.sw(6)=%F;
end
if env_or_empty("AI_FREE_W")=="1" then
    experiment.sw(6)=%T;
end
if env_or_empty("AI_FREE_A")=="0" then
    experiment.sw(7)=%F;
end
if env_or_empty("AI_FREE_A")=="1" then
    experiment.sw(7)=%T;
end
if env_or_empty("AI_FREE_Y0")=="0" then
    experiment.sw(10)=%F;
end
if env_or_empty("AI_FREE_Y0")=="1" then
    experiment.sw(10)=%T;
end
if env_or_empty("AI_THIRD_ORDER")=="0" then
    experiment.third_order_calc=%F;
end
fit_position=env_or_empty("AI_POSITION");
if fit_position<>"" then
    experiment.position=evstr(fit_position);
    experiment.jposition=experiment.position;
end
if env_or_empty("AI_NUM_SPINS")=="3" then
    experiment.jposition=experiment.position;
end
init_u_pos=env_or_empty("AI_INIT_U1");
if init_u_pos<>"" then
    experiment.atom(experiment.position).U=evstr(init_u_pos);
end
init_jrho_pos=env_or_empty("AI_INIT_JRHO1");
if init_jrho_pos<>"" then
    experiment.atom(experiment.position).J=evstr(init_jrho_pos);
end
init_w_pos=env_or_empty("AI_INIT_W1");
if init_w_pos<>"" then
    experiment.atom(experiment.position).w=evstr(init_w_pos);
end
fit_xrange=env_or_empty("AI_XRANGE");
if fit_xrange<>"" then
    experiment.xrange=evstr(fit_xrange);
    %Data_trunc=%Data(find(abs(%Data(:,1))<=experiment.xrange),:);
    %weight=ones(size(%Data_trunc,1),1);
end
if fit_xrange<>"" & env_or_empty("AI_WEIGHT_NEGPEAK")=="1" then
    idx=find(%Data_trunc(:,1)>=-2.8 & %Data_trunc(:,1)<=-1.6);
    %weight(idx)=5;
    idx=find(%Data_trunc(:,1)>=-4.5 & %Data_trunc(:,1)<=-3.5);
    %weight(idx)=2;
end
if env_or_empty("AI_WEIGHT_A1_SMALL")=="1" then
    %weight=ones(size(%Data_trunc,1),1);
    idx=find(%Data_trunc(:,1)>=-2.25 & %Data_trunc(:,1)<=-1.75);
    %weight(idx)=14;
    idx=find(%Data_trunc(:,1)>=-2.8 & %Data_trunc(:,1)<-2.25);
    %weight(idx)=4;
    idx=find(%Data_trunc(:,1)>-1.75 & %Data_trunc(:,1)<=-1.35);
    %weight(idx)=4;
    idx=find(%Data_trunc(:,1)>=2.75 & %Data_trunc(:,1)<=3.45);
    %weight(idx)=4;
end
if env_or_empty("AI_WEIGHT_A1_NEGPOS")=="1" then
    %weight=ones(size(%Data_trunc,1),1);
    idx=find(%Data_trunc(:,1)>=-2.12 & %Data_trunc(:,1)<=-1.82);
    %weight(idx)=35;
    idx=find(%Data_trunc(:,1)>=-2.45 & %Data_trunc(:,1)<-2.12);
    %weight(idx)=8;
    idx=find(%Data_trunc(:,1)>-1.82 & %Data_trunc(:,1)<=-1.45);
    %weight(idx)=8;
    idx=find(%Data_trunc(:,1)>=2.8 & %Data_trunc(:,1)<=3.45);
    %weight(idx)=3;
end
if env_or_empty("AI_WEIGHT_TRIMER_EDGE")=="1" then
    %weight=ones(size(%Data_trunc,1),1);
    idx=find(%Data_trunc(:,1)>=-2.4 & %Data_trunc(:,1)<=-0.8);
    %weight(idx)=3;
    idx=find(%Data_trunc(:,1)>=0.8 & %Data_trunc(:,1)<=3.6);
    %weight(idx)=12;
    idx=find(%Data_trunc(:,1)>3.6 & %Data_trunc(:,1)<=5.0);
    %weight(idx)=4;
end
if env_or_empty("AI_WEIGHT_FEATURES")=="1" then
    %weight=ones(size(%Data_trunc,1),1);
    cguess=abs(experiment.heisenberg_coupling);
    feature_pos=cguess;
    feature_pos_env=env_or_empty("AI_FEATURE_POS");
    if feature_pos_env<>"" then feature_pos=abs(evstr(feature_pos_env)); end
    peak_fraction=0.16;
    center_fraction=0.10;
    shoulder_fraction=0.28;
    peak_weight=5;
    center_weight=4;
    shoulder_weight=2;
    weight_value=env_or_empty("AI_FEATURE_PEAK_FRACTION");
    if weight_value<>"" then peak_fraction=evstr(weight_value); end
    weight_value=env_or_empty("AI_FEATURE_CENTER_FRACTION");
    if weight_value<>"" then center_fraction=evstr(weight_value); end
    weight_value=env_or_empty("AI_FEATURE_SHOULDER_FRACTION");
    if weight_value<>"" then shoulder_fraction=evstr(weight_value); end
    weight_value=env_or_empty("AI_FEATURE_PEAK_WEIGHT");
    if weight_value<>"" then peak_weight=evstr(weight_value); end
    weight_value=env_or_empty("AI_FEATURE_CENTER_WEIGHT");
    if weight_value<>"" then center_weight=evstr(weight_value); end
    weight_value=env_or_empty("AI_FEATURE_SHOULDER_WEIGHT");
    if weight_value<>"" then shoulder_weight=evstr(weight_value); end
    peak_half=max(0.35,peak_fraction*feature_pos);
    center_half=max(0.35,center_fraction*feature_pos);
    idx=find(abs(abs(%Data_trunc(:,1))-feature_pos)<=peak_half);
    %weight(idx)=peak_weight;
    idx=find(abs(%Data_trunc(:,1))<=center_half);
    %weight(idx)=center_weight;
    shoulder_half=max(0.65,shoulder_fraction*feature_pos);
    idx=find(abs(abs(%Data_trunc(:,1))-feature_pos)<=shoulder_half & ...
        abs(abs(%Data_trunc(:,1))-feature_pos)>peak_half);
    %weight(idx)=shoulder_weight;
end
custom_weight_text=env_or_empty("AI_CUSTOM_WEIGHT_WINDOWS");
if custom_weight_text<>"" then
    custom_weight_windows=evstr(custom_weight_text);
    if size(custom_weight_windows,2)<>3 then
        error("AI_CUSTOM_WEIGHT_WINDOWS must have three columns: xmin, xmax, weight");
    end
    for weight_row=1:size(custom_weight_windows,1)
        idx=find(%Data_trunc(:,1)>=custom_weight_windows(weight_row,1) & ...
            %Data_trunc(:,1)<=custom_weight_windows(weight_row,2));
        %weight(idx)=custom_weight_windows(weight_row,3);
    end
end

calc.spec = [];
calc.states= [];
calc.occ = [];
calc.lifetime = [];
calc.entropy = [];
calc.giantspin= [];
calc.negativity= [];
calc.spec_raw=[];
old_k=[];

function write_result(filename,k,fopt)
    global experiment calc;
    fd=mopen(filename,"wt");
    mfprintf(fd,"fopt %f\n",fopt);
    mfprintf(fd,"k ");
    for ii=1:size(k,"*")
        mfprintf(fd,"%f ",k(ii));
    end
    mfprintf(fd,"\n");
    pos=experiment.position;
    mfprintf(fd,"position %d\n",pos);
    mfprintf(fd,"Upos %f\n",experiment.atom(pos).U);
    mfprintf(fd,"Jrho_pos %f\n",experiment.atom(pos).J);
    mfprintf(fd,"wpos %f\n",experiment.atom(pos).w);
    mfprintf(fd,"A %f\n",experiment.A);
    mfprintf(fd,"b %f\n",experiment.b);
    mfprintf(fd,"x0 %f\n",experiment.x0);
    mfprintf(fd,"y0 %f\n",experiment.y0);
    mfprintf(fd,"T_meV %f\n",experiment.T);
    mfprintf(fd,"lt_meV %f\n",experiment.lt);
    mfprintf(fd,"C %f\n",experiment.heisenberg_coupling);
    if calc.spec_raw<>[] then
        y=experiment.A*(real(calc.spec_raw(:,2))+experiment.y0+experiment.b*calc.spec_raw(:,1));
        for ii=1:size(calc.spec_raw,1)
            mfprintf(fd,"fit %f %f\n",calc.spec_raw(ii,1)+experiment.x0,y(ii));
        end
    end
    mclose(fd);
endfunction

function write_gui_export(filename)
    global experiment calc;
    tempexp=[];
    fields=fieldnames(experiment);
    for fieldIndex=1:size(fields,1)
        if fields(fieldIndex)<>"Eigenvec" & fields(fieldIndex)<>"Eigenval" then
            execstr("tempexp."+fields(fieldIndex)+"=experiment."+fields(fieldIndex));
        end
    end
    text=string(tempexp);
    atomlink=experiment.atom;
    if size(atomlink,2)>1 then
        tmp=experiment.matrix;
        text=[text;'Heisenberg Matrixvalues:'];
        for columnIndex=1:size(tmp,2)
            str=' ';
            for rowIndex=1:size(tmp,1)
                str=str+tmp(rowIndex,columnIndex)+';';
            end
            text=[text;str];
        end
        tmp=experiment.matrixDM;
        text=[text;'DM Matrixvalues:'];
        for columnIndex=1:size(tmp,2)
            str=' ';
            for rowIndex=1:size(tmp,1)
                str=str+tmp(rowIndex,columnIndex)+';';
            end
            text=[text;str];
        end
    end
    for atomIndex=1:size(atomlink,2)
        text=[text;'Spin '+string(atomIndex)+':'];
        text=[text;string(atomlink(atomIndex))];
    end
    data=calc.spec;
    savedat(filename,data($:-1:1,1),real(data($:-1:1,2:$)),text);
endfunction

bound_value=env_or_empty("AI_U_MIN");
if bound_value<>"" then minvalues1(4)=evstr(bound_value); end
bound_value=env_or_empty("AI_U_MAX");
if bound_value<>"" then maxvalues1(4)=evstr(bound_value); end
bound_value=env_or_empty("AI_JRHO_MIN");
if bound_value<>"" then minvalues1(5)=evstr(bound_value); end
bound_value=env_or_empty("AI_JRHO_MAX");
if bound_value<>"" then maxvalues1(5)=evstr(bound_value); end
bound_value=env_or_empty("AI_W_MIN");
if bound_value<>"" then minvalues1(6)=evstr(bound_value); end
bound_value=env_or_empty("AI_W_MAX");
if bound_value<>"" then maxvalues1(6)=evstr(bound_value); end
bound_value=env_or_empty("AI_A_MIN");
if bound_value<>"" then minvalues1(7)=evstr(bound_value); end
bound_value=env_or_empty("AI_A_MAX");
if bound_value<>"" then maxvalues1(7)=evstr(bound_value); end
bound_value=env_or_empty("AI_B_MIN");
if bound_value<>"" then minvalues1(8)=evstr(bound_value); end
bound_value=env_or_empty("AI_B_MAX");
if bound_value<>"" then maxvalues1(8)=evstr(bound_value); end
bound_value=env_or_empty("AI_X0_MIN");
if bound_value<>"" then minvalues1(9)=evstr(bound_value); end
bound_value=env_or_empty("AI_X0_MAX");
if bound_value<>"" then maxvalues1(9)=evstr(bound_value); end
bound_value=env_or_empty("AI_Y0_MIN");
if bound_value<>"" then minvalues1(10)=evstr(bound_value); end
bound_value=env_or_empty("AI_Y0_MAX");
if bound_value<>"" then maxvalues1(10)=evstr(bound_value); end
bound_value=env_or_empty("AI_T_MIN");
if bound_value<>"" then minvalues1(11)=evstr(bound_value); end
bound_value=env_or_empty("AI_T_MAX");
if bound_value<>"" then maxvalues1(11)=evstr(bound_value); end

[k0,fitmin,fitmax]=exp_to_k(experiment);
c_min=env_or_empty("AI_C_MIN");
c_max=env_or_empty("AI_C_MAX");
if size(experiment.atom,2)>1 & experiment.sw(13) then
    cidx=1;
    if experiment.sw(1) then cidx=cidx+1; end
    if experiment.sw(2) then cidx=cidx+1; end
    if experiment.sw(3) then cidx=cidx+1; end
    if c_min<>"" then fitmin(cidx)=evstr(c_min); end
    if c_max<>"" then fitmax(cidx)=evstr(c_max); end
end
fst=create_f_fit(experiment);
selected_xrange=experiment.xrange;
execstr(fst);
fstart=f_fit(k0);

function [fk,dfk,ind]=costfkt(k,ind)
    global itcount;
    fk=f_fit(k);
    dfk=numderivative(f_fit,k);
    itcount=itcount+1;
endfunction

itcount=0;
max_calls=80;
max_iter=40;
max_calls_env=env_or_empty("AI_MAX_CALLS");
if max_calls_env<>"" then
    max_calls=evstr(max_calls_env);
end
max_iter_env=env_or_empty("AI_MAX_ITER");
if max_iter_env<>"" then
    max_iter=evstr(max_iter_env);
end
[fopt,k,gopt]=optim(costfkt, "b", fitmin, fitmax, k0, "ar", max_calls, max_iter, imp=1);
fopt=f_fit(k);
experiment=k_to_exp(k,experiment);
experiment.xrange=selected_xrange;
write_result(output_file,k,fopt);

// The optimizer evaluates the model only at experimental bias points. Recompute
// the converged model on a dense grid so exported curves and saved GUI programs
// remain smooth even when the input data were physically truncated or sparse.
dense_points=1001;
dense_points_env=env_or_empty("AI_DENSE_OUTPUT_POINTS");
if dense_points_env<>"" then dense_points=max(101,round(evstr(dense_points_env))); end
dense_x=linspace(-selected_xrange,selected_xrange,dense_points)';
%Data_trunc=[dense_x,zeros(dense_points,1)];
%weight=ones(dense_points,1);
calc.spec_raw=[];
old_k=k;
dense_dummy=f_fit(k);
experiment=k_to_exp(k,experiment);
experiment.xrange=selected_xrange;
calc.spec=[calc.spec_raw(:,1)+experiment.x0, ...
    experiment.A*(real(calc.spec_raw(:,2))+experiment.y0+experiment.b*calc.spec_raw(:,1))];
program_file=env_or_empty("AI_PROGRAM_OUTPUT");
if program_file<>"" then
    save(program_file,"experiment","calc");
end
export_file=env_or_empty("AI_EXPORT_OUTPUT");
if export_file<>"" then
    write_gui_export(export_file);
end
disp("fstart="+string(fstart));
disp("fopt="+string(fopt));
disp("k="+string(k));
quit;

