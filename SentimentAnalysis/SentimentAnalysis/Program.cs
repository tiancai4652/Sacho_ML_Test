using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;

namespace SentimentAnalysis
{
    class Program
    {

        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "ModelData", "yelp_labelled.txt");

        static void Main(string[] args)
        {
            //创建一个新的 ML.NET 环境
            MLContext mlContext = new MLContext();
            //加载数据，获取训练数据和测试数据
            TrainTestData splitDataView = LoadData(mlContext);
            //用训练数据训练出一个模型
            ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);
            //用测试测试数据评估模型
            Evaluate(mlContext, model, splitDataView.TestSet);
            //对单个项目使用模型
            UseModelWithSingleItem(mlContext, model);
            //使用模型进行批量预测
            UseModelWithBatchItems(mlContext, model);
        }

        /// <summary>
        /// 加载数据。
        /// 准备模型时，使用部分数据集来训练它，并使用部分数据集来测试模型的准确性。
        /// 将加载的数据集拆分为训练数据集和测试数据集。
        /// 返回拆分的训练数据集和测试数据集。
        /// </summary>
        /// <param name="mlContext"></param>
        /// <returns></returns>
        public static TrainTestData LoadData(MLContext mlContext)
        {
            //用于定义数据架构并读取文件
            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);
            //将加载的数据集拆分为训练数据集和测试数据集,默认值为 10%，在本例中使用 20%，以评估更多数据。
            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            return splitDataView;
        }

        /// <summary>
        /// 提取并转换数据。
        /// 定型模型。
        /// 根据测试数据预测情绪。
        /// 返回模型。
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="splitTrainSet"></param>
        /// <returns></returns>
        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
        {
            ///定义评估器
            var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
        .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));
            Console.WriteLine("=============== Create and Train the Model ===============");
            ///使模型适应 splitTrainSet 数据
            var model = estimator.Fit(splitTrainSet);
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();
            return model;
        }

        /// <summary>
        /// 加载测试数据集。
        /// 创建 BinaryClassification 计算器。
        /// 评估模型并创建指标。
        /// 显示指标
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="model"></param>
        /// <param name="splitTestSet"></param>
        public static void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
            IDataView predictions = model.Transform(splitTestSet);
            ///评估模型
            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
            ///显示评估的指标信息
            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            //模型准确度
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            //模型对正面类和负面类进行正确分类的置信度。 应该使 AreaUnderRocCurve 尽可能接近 1
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
            //查准率和查全率之间的平衡关系的度量值。 应该使 F1Score 尽可能接近 1。
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");
        }

        /// <summary>
        /// 创建测试数据。
        /// 结合测试数据和模型进行预测。
        /// 显示预测结果。
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="model"></param>
        private static void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
        {
            PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
            SentimentData sampleStatement = new SentimentData
            {
                SentimentText = "This was a very bad steak"
            };
            var resultPrediction = predictionFunction.Predict(sampleStatement);
            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

            Console.WriteLine();
            Console.WriteLine($"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");

            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();
        }

        /// <summary>
        /// 创建批处理测试数据。
        /// 根据测试数据预测情绪。
        /// 结合测试数据和预测进行报告。
        /// 显示预测结果。
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="model"></param>
        public static void UseModelWithBatchItems(MLContext mlContext, ITransformer model)
        {
            //创建测试数据
            IEnumerable<SentimentData> sentiments = new[]{

    new SentimentData
    {
        SentimentText = "This was a horrible meal"
    },
    new SentimentData
    {
        SentimentText = "I love this spaghetti."
    }
            };


            IDataView batchComments = mlContext.Data.LoadFromEnumerable(sentiments);

            IDataView predictions = model.Transform(batchComments);

            // Use model to predict whether comment data is Positive (1) or Negative (0).
            IEnumerable<SentimentPrediction> predictedResults = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);
            Console.WriteLine();

            Console.WriteLine("=============== Prediction Test of loaded model with multiple samples ===============");

            foreach (SentimentPrediction prediction in predictedResults)
            {
                Console.WriteLine($"Sentiment: {prediction.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");

            }
            Console.WriteLine("=============== End of predictions ===============");
        }
    }
}
