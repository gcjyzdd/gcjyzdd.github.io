---
layout: post
date:   2018-03-02 9:51
categories: Work Simulink
---


## Rebuild PreScan in InitFcn

<div style="text-align:center"><img src ='{{site.baseurl}}/assets/tass/initFcn_rebuild_TA.png' /></div>

The following code shows how to rebuild pex files and regenerate simulink mdl files. The idea is:

* get value of a constant block
* set the test automation variable of pex file and rebuild pex file
* error checking
* regenerate mdl files

In this example, we'd like to set acceleration of a target vehicle `host_acc_x` (forget the variable name). 

```matlab
% get model name
[direc,mname]=fileparts(get_param(bdroot, 'FileName'));
% get full path of <experimentName>.pex from <experimentName>_cs.mdl
expPathname = [direc , '\' , mname(1:length(mname)-3) , '.pex'];
% get block name
bname = [mname, '/TEST Acc No. {1,2}'];
% get blocks value
bvalue = str2num(get_param(bname,'Value'));

% % get block name
% bname = [mname, '/TEST Acc No. {1,2}'];
% % get blocks value
% bvalue2 = str2num(get_param(bname,'Value'));

% checking correctness of a test number
if ( bvalue < 0 ) || ( bvalue > 2 ) 
    error('Incorrect test No. Please enter: 1 to 2')
end

offsets = [28.44 28.44+40-12];
host_acc_x = [-2 -6];

status = system(['PreScan.CLI.exe -load ',expPathname,...
    ' -set host_acc_x=', num2str(offsets(bvalue)*0.95),...
    ' -build']);
if status ~= 0
    error('The experiment was not rebuild properly. Please open PreScan GUI before next try.')
end

% regenerate the model
generate_callback(bdroot);
```

## Offset an object in Simulink

### Use trajectory file

<div style="text-align:center"><img width='100%' src ='{{site.baseurl}}/assets/tass/offset_by_traj_swtich.png' /></div>

In this example, our goal is to shift a target vehicle. We can achieve it by TA the distance of trajectory of the target vehicle in PreScan. That needs to rebuild in PreScan. Apart from that, we could also change the offset directly in Simulink, without rebuild and regeneration.

As shown in the figure above, a multi-port switch block is added and let the user to select which trajectory should be adoptted.

### Use path follower

<div style="text-align:center"><img width='100%' src ='{{site.baseurl}}/assets/tass/offset_by_path_follower.png' /></div>
