----------------------------------------------------------------------
require 'torch'
require 'image'
require 'nn'
matio = require 'matio'
----------------------------------------------------------------------

index_train = torch.randperm(10000)

-- training/test size
trsize = 9000
valsize = 1000
tesize = 8000
trainData = {}
trainData.data = torch.Tensor(trsize, 3,96,96)
trainData.labels = torch.Tensor(trsize)
trainData.size = trsize
valData = {}
valData.data = torch.Tensor(valsize, 3, 96, 96)
valData.labels = torch.Tensor(valsize)
valData.size = valsize
----------------------------------------------------------------------
print '==> loading dataset'

train = matio.load('train.mat')
train_x = train.X
train_x = torch.reshape(train_x, 5000, 3,96,96):transpose(3,4):float()
train_y = train.y:float()
test = matio.load('test.mat')
test_x = test.X
test_x = torch.reshape(test_x, 8000, 3,96,96):transpose(3,4):float()
test_y = test.y:float()
print '==>augmentation'
-- horizontally flip the picture
train_x_aug = torch.Tensor(10000, 3, 96, 96)
train_y_aug = torch.repeatTensor(train_y,2,1)[{{},1}]
for i = 1, 5000 do
   x = train_x[i]
   train_x_aug[i] = x
   train_x_aug[5000 + i] = image.hflip(x)
end

-- Train Valid Split
print("==> Split training/validation sets")
for i =1,trsize do
	trainData.data[i] = train_x_aug[index_train[i]]
	trainData.labels[i] = train_y_aug[index_train[i]]
	collectgarbage()
end
for i =1,valsize do
	valData.data[i] = train_x_aug[index_train[i+trsize]]
	valData.labels[i] = train_y_aug[index_train[i+trsize]]
	collectgarbage()
end
testData = {
   data = test_x,
   labels = test_y[{{},1}],
   size = tesize 
}
print(trainData.labels:size())
print(valData.labels:size())
print(testData.labels:size())
----------------------------------------------------------------------
print '==> preprocessing data'

-- Convert all images to YUV
print '==> preprocessing data: colorspace RGB -> YUV'
for i = 1,trainData.size do
   trainData.data[i] = image.rgb2yuv(trainData.data[i])
end
for i = 1,valData.size do
   valData.data[i] = image.rgb2yuv(valData.data[i])
end
for i = 1,testData.size do
   testData.data[i] = image.rgb2yuv(testData.data[i])
end

-- Name channels for convenience
channels = {'y','u','v'}

-- Normalize each channel, and store mean/std
-- per channel. These values are important, as they are part of
-- the trainable parameters. At test time, test data will be normalized
-- using these values.
print '==> preprocessing data: normalize each feature (channel) globally'
mean = {}
std = {}
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   mean[i] = trainData.data[{ {},i,{},{} }]:mean()
   std[i] = trainData.data[{ {},i,{},{} }]:std()
   trainData.data[{ {},i,{},{} }]:add(-mean[i])
   trainData.data[{ {},i,{},{} }]:div(std[i])
end

-- Normalize test/validation data, using the training means/stds
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   testData.data[{ {},i,{},{} }]:add(-mean[i])
   testData.data[{ {},i,{},{} }]:div(std[i])
   valData.data[{ {},i,{},{} }]:add(-mean[i])
   valData.data[{ {},i,{},{} }]:div(std[i])
end

-- Local normalization
print '==> preprocessing data: normalize all three channels locally'

-- Define the normalization neighborhood:
neighborhood = image.gaussian1D(13)
print '------'
-- Define our local normalization operator (It is an actual nn module, 
-- which could be inserted into a trainable model):
normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()
print '--------'
-- Normalize all channels locally:
for c in ipairs(channels) do
   for i = 1,trsize do
      trainData.data[{ i,{c},{},{} }] = normalization:forward(trainData.data[{ i,{c},{},{} }]:float())
   end
   for i = 1,valsize do
      valData.data[{ i,{c},{},{} }] = normalization:forward(valData.data[{ i,{c},{},{} }]:float())
   end
   for i = 1,tesize do
      testData.data[{ i,{c},{},{} }] = normalization:forward(testData.data[{ i,{c},{},{} }]:float())
   end
end

----------------------------------------------------------------------
print '==> verify statistics'

-- It's always good practice to verify that data is properly
-- normalized.

for i,channel in ipairs(channels) do
   trainMean = trainData.data[{ {},i }]:mean()
   trainStd = trainData.data[{ {},i }]:std()

   valMean = valData.data[{ {},i }]:mean()
   valStd = valData.data[{ {},i }]:std()

   testMean = testData.data[{ {},i }]:mean()
   testStd = testData.data[{ {},i }]:std()

   print('training data, '..channel..'-channel, mean: ' .. trainMean)
   print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)

   print('validation data, '..channel..'-channel, mean: ' .. valMean)
   print('validation data, '..channel..'-channel, standard deviation: ' .. valStd)

   print('test data, '..channel..'-channel, mean: ' .. testMean)
   print('test data, '..channel..'-channel, standard deviation: ' .. testStd)
end

----------------------------------------------------------------------
-- print '==> visualizing data'
-- 
-- -- Visualization is quite easy, using gfx.image().
-- 
-- if opt.visualize then
--    first256Samples_y = trainData.data[{ {1,256},1 }]
--    first256Samples_u = trainData.data[{ {1,256},2 }]
--    first256Samples_v = trainData.data[{ {1,256},3 }]
--    gfx.image(first256Samples_y, {legend='Y'})
--    gfx.image(first256Samples_u, {legend='U'})
--    gfx.image(first256Samples_v, {legend='V'})
-- end
