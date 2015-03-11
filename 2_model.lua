require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers

----------------------------------------------------------------------

print '==> define parameters'
-- 10-class problem
noutputs = 10

-- input dimensions
nfeats = 3
width = 96
height = 96
ninputs = nfeats*width*height

-- number of hidden units (for MLP only):
nhiddens = ninputs / 2

-- hidden units, filter sizes (for ConvNet only):
nstates = {128,128,256,256}
filtsize = {5,8}
poolsize = 2
normkernel = image.gaussian1D(7)

model = nn.Sequential()

-- stage 1 : filter bank -> squashing -> normalization
model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize[1], filtsize[1]))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

-- stage 2 : filter bank -> squashing -> L2 pooling -> normalization 
model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize[1], filtsize[1]))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

-- stage 3 : filter bank -> squashing -> L2 pooling -> normalization
model:add(nn.SpatialConvolutionMM(nstates[2], nstates[3], filtsize[2], filtsize[2]))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

-- stage 4 : standard 2-layer neural network
model:add(nn.View(nstates[3]*7*7))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(nstates[3]*7*7, nstates[4]))
model:add(nn.ReLU())
model:add(nn.Linear(nstates[4], noutputs))

-----------------------------------------------------------------
print '==>here is the model'
print(model)

----------------------------------------------------------------------
------ Visualization is quite easy, using gfx.image().
----
----if opt.visualize then
----   if opt.model == 'convnet' then
----      print '==> visualizing ConvNet filters'
----      gfx.image(model:get(1).weight, {zoom=2, legend='L1'})
----      gfx.image(model:get(5).weight, {zoom=2, legend='L2'})
----   end
----end
