---
layout: post
title: "SDC in PreScan: Step 2-- Keyboard Inputs in Simulink"
Date: 2017-01-03 09:03
---

# Goals/Steps:

1. Build a deep learning server on linux machine(or windows) using Python
2. Build an interface on a Simulink model. Take keyboard inputs to control vehicle
3. Use keyboard inputs to control vehicle and use three parallel cameras to generate proper data set
4. Use the data set from step 3 to train the neural network
5. Send image from Simulink to the server using tcp/udp socket, and receive steering angles from the server

# Introduction

To get real time keyboard inputs when running Simulink, we need to create a figure and set its callbacks such
 that the keyboard status is stored in `guidata`. 

# Setup S-Function

## Input and Output ports

To drive a car, at least the steering angle and gas are needed. Hence, we use left, right, up, and down arrow to represent
steering and acceleration. In the `S-Function`, there should two output ports and no input ports. The following is the first
part of `S-Function` of the `Keyboard` block:

```
function sfun_keyboard_input_v1_2b(block)
% modified from sfun_keyboard_input_v1_01.m of Marc Compere by Emanuele Ruffaldi
% created : 17 June 2003
% modified: 20 June 2003
% created: 19 May 2009 => Level 2 and terminate
%

% Level-2 M file S-Function with keyboard
%   Copyright 1990-2004 The MathWorks, Inc.
%   $Revision: 1.1.6.1 $

%
%
% Updates:
% Detect multiple directional key inputs
% modified by Changjie Guan<changjie.guan@tassinternational.com>
% 2018 Jan 3

setup(block);

%endfunction

function setup(block)

block.NumDialogPrms  = 0;
block.NumInputPorts  = 0;
block.NumOutputPorts = 3;

block.OutputPort(1).Dimensions       = 1;
block.OutputPort(1).SamplingMode = 'Sample';
block.OutputPort(1).DatatypeID  = 0;
block.OutputPort(2).Dimensions       = 1;
block.OutputPort(2).SamplingMode = 'Sample';
block.OutputPort(3).DatatypeID  = 0;
block.OutputPort(3).Dimensions       = 1;
block.OutputPort(3).SamplingMode = 'Sample';
block.OutputPort(3).DatatypeID  = 0;
block.SampleTimes = [-1, 0];
%end

%% Register methods
%  block.RegBlockMethod('CheckParameters', @CheckPrms);
block.RegBlockMethod('PostPropagationSetup',    @DoPostPropSetup);
block.RegBlockMethod('InitializeConditions',    @InitConditions);
block.RegBlockMethod('Outputs',                 @Output);
block.RegBlockMethod('Update',                  @Update);
block.RegBlockMethod('Terminate',                 @Terminate);
%endfunction
```

## Setup States

To store the data of two output ports, two `Dwork` are added as field members to the block. As mentioned above, 
a figure is required to receive keyboard inputs and an additional `Dwork` is used. The settings is as below:

```
function DoPostPropSetup(block)

%% Setup Dwork
block.NumDworks = 3;
block.Dwork(1).Name = 'key';
block.Dwork(1).Dimensions      = 1;
block.Dwork(1).DatatypeID      = 0;
block.Dwork(1).Complexity      = 'Real';
block.Dwork(1).UsedAsDiscState = true;

block.Dwork(2).Name = 'fig';
block.Dwork(2).Dimensions      = 1;
block.Dwork(2).DatatypeID      = 0;
block.Dwork(2).Complexity      = 'Real';
block.Dwork(2).UsedAsDiscState = false;

block.Dwork(3).Name = 'new';
block.Dwork(3).Dimensions      = 1;
block.Dwork(3).DatatypeID      = 0;
block.Dwork(3).Complexity      = 'Real';
block.Dwork(3).UsedAsDiscState = true;
%endfunction
```

## Initialization

In the initialization function, we mainly do two tasks: set up the figure and initialize `guidata`. We store five values
in `guidata`:

```
    % set guidata
    myhandles = guihandles(handle.figure);
    myhandles.left = 0;
    myhandles.right = 0;
    myhandles.up = 0;
    myhandles.down = 0;
    myhandles.record = 0;
    
    guidata(handle.figure, myhandles)
```

The `left`, `right`, `up`, and `down` store the keyboard status of `leftarrow`, etc. And the field `record` is to indicate whether
the present data is adoppted. 

For the figure, we set two callbacks for `KeyPressFcn` and `KeyReleaseFcn`, which will be explained in the next section.

The complete code of initialization function is as below:

```
function InitConditions(block)

%% Initialize Dwork
block.Dwork(1).Data = 0;
handle.figure=findobj('type','figure','Tag','keyboard input figure');

if isempty(handle.figure)
    % 'position' args -> [left, bottom, width, height]
    handle.figure=figure('position',[50 50 400 200],...
        'WindowStyle','Modal',...
        'Name','Keyboard Input',...
        'Color',get(0,'DefaultUicontrolBackgroundColor')); %,...
    %'HandleVisibility','callback');
    %handle.figure=figure('position',[800 620 400 300]);
    %handle.figure=figure('position',[800 620 400 300],'WindowButtonDownFcn',@myCallback)
    %handle.figure=figure('position',[800 620 400 300],'WindowButtonMoveFcn',@myCallback_move,'WindowButtonDownFcn',@myCallback_clickdown)
    set(handle.figure,'Tag','keyboard input figure');

    % Make the ON button (position args->[left bottom width height])
    handle.recordbutton = uicontrol(handle.figure,...
        'Style','pushbutton',...
        'Units','characters',...
        'Position',[5 9 46 5],...
        'BackgroundColor', [1 0 0],...
        'String','Record Data',...
        'FontSize', 24,...
        'FontWeight', 'Bold',...
        'ForegroundColor',[1 1 1],...
        'Callback',{@myRecordData,handle});
    
    % Make the OFF button (position args->[left bottom width height])
    handle.offbutton = uicontrol(handle.figure,...
        'Style','pushbutton',...
        'Units','characters',...
        'Position',[5 5 46 2],...
        'String','Disable exclusive figure-keyboard input',...
        'Callback',{@turn_modal_off,handle});
    
    % Make the ON button (position args->[left bottom width height])
    handle.onbutton = uicontrol(handle.figure,...
        'Style','pushbutton',...
        'Units','characters',...
        'Position',[5 1 46 2],...
        'String','Re-enable exclusive figure-keyboard input',...
        'Callback',{@turn_modal_on,handle});
    
    set(handle.figure, 'KeyPressFcn', @myKeyPress);
    set(handle.figure, 'KeyReleaseFcn', @myKeyRelease);
    
    % set guidata
    myhandles = guihandles(handle.figure);
    myhandles.left = 0;
    myhandles.right = 0;
    myhandles.up = 0;
    myhandles.down = 0;
    myhandles.record = 0;
    
    guidata(handle.figure, myhandles)
    
else % reset the figure to 'modal' to continue accepting keyboard input
    set(handle.figure,'WindowStyle','Modal')
end
if strcmp(class(handle.figure),'double')
    block.Dwork(2).Data = handle.figure;
else
    block.Dwork(2).Data = handle.figure.Number;
end
```

## Set Callbacks for the figure

The method `get(handle.figure,'CurrentCharacter')` that captures keyboard inputs can only save the last event. For instance,
if two keys are pressed only the later pressed can be recorded. In order to record multiple keyboard inputs, we resort to `KeyReleaseFcn`
to save status of pressed keys.

The implmentation of those callbacks are as follow:

```
function myKeyPress(~, event)

myhandles = guidata(gcbo);
switch event.Key
    case 'leftarrow'
        myhandles.left = 1;
    case 'rightarrow'
        myhandles.right = 1;
    case 'uparrow'
        myhandles.up = 1;
    case 'downarrow'
        myhandles.down = 1;
    otherwise
end

guidata(gcbo,myhandles)

function myKeyRelease(~, event)

myhandles = guidata(gcbo);
switch event.Key
    case 'leftarrow'
        myhandles.left = 0;
    case 'rightarrow'
        myhandles.right = 0;
    case 'uparrow'
        myhandles.up = 0;
    case 'downarrow'
        myhandles.down = 0;
    otherwise
end

guidata(gcbo,myhandles)
```

And below is the code to update `record` status and refresh the appearience of the pushbuttom:

```
% Callback for recording data
function myRecordData(~, event, handle)

myhandles = guidata(gcbo);
myhandles.record = ~myhandles.record;
guidata(gcbo,myhandles)

if myhandles.record
    set(gcbo, 'BackgroundColor', [0 1 0])
else
    set(gcbo, 'BackgroundColor', [1 0 0])
end
```

## Output and Update

The implementation of `Output` and `Update` method is straightforward and the code is pasted here:

```
function Output(block)

handle.figure = block.Dwork(2).Data;
myhandles = guidata(handle.figure);

% block.OutputPort(1).Data = block.Dwork(1).Data;
% block.OutputPort(2).Data = block.Dwork(3).Data;
block.OutputPort(1).Data = myhandles.left - myhandles.right;
block.OutputPort(2).Data = myhandles.up - myhandles.down;
block.OutputPort(3).Data = double(myhandles.record);

%endfunction

function Update(block)

handle.figure = block.Dwork(2).Data;
%handle = get(handle.figure,'userdata');


current_char=get(handle.figure,'CurrentCharacter'); % a single character, like 'b'

% update the grahics object
%set(handle.point,'Xdata',[x(1) x(1)],'Ydata',[x(2) x(2)],'Zdata',[-1 +1]);


% conditionally update the (numeric) state
if ~isempty(current_char)
    block.Dwork(1).Data =double(current_char);
    block.Dwork(3).Data = 1;
    % reset 'CurrentCharacter' so if user lifts up from key, this is noticed
    set(handle.figure,'CurrentCharacter',char(0)) % the plus key is the only key that may be
else
    block.Dwork(3).Data = 0;
end
```

**Note**: remember to save data after updating 'myhandles'.

## Termination

Destroy the figure when the simulation is finished:

```
function Terminate(block)
handle.figure = block.Dwork(2).Data;
if handle.figure ~= 0
    close (handle.figure);
end
```

## Demo

Here is a screenshot of using this block:

<div style="text-align:center"><img src="{{site.baseurl}}/assets/SDC-PreScan/keyboard_inputs.png" /></div>

# Summary

In this post, we created a simulation block using `Matlab S-Function Level 2` that takes directional keyboard inputs and outputs steering direction and acceleration.
The follwing techniques are applied:

* set input and output ports, including dimension, data types, sample time
* set callbacks for figures
* set `guidata` to store user data

