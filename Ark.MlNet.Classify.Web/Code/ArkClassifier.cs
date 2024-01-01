using ark.net.util;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Ark.MlNet.Classify
{
    public class ClassifyModel
    {
        [LoadColumn(0)]
        public string Text { get; set; }
        [LoadColumn(1)]
        public string Tag { get; set; }
    }
    public enum MyTrainerStrategy : int
    {
        SdcaMultiClassTrainer = 1,
        OVAAveragedPerceptronTrainer = 2
    };
    public class ArkClassifier
    {
        public static void Predict()
        {
            var mlContext = new MLContext(seed: 1);
            DataViewSchema modelSchema;
            ITransformer trainedModel = mlContext.Model.Load("./MLModels/Jobss_v1_Model.zip", out modelSchema);
            //var issue = new ClassifyModel() { KeySkill = "WebSockets communication is slow in my machine", RoleCategory = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.." };
            // Create prediction engine related to the loaded trained model
            //var predEngine = mlContext.Model.CreatePredictionEngine<ResumeLabel, ResumePrediction>(trainedModel);
            //Score
            //var prediction = predEngine.Predict(issue);
            //Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.JobTitle}, Score: {prediction.Score}  ===============");
        }
        public static void BuildAndTrainModel(MyTrainerStrategy selectedStrategy, string path)
        {
            var mlContext = new MLContext(seed: 1);
            var trainingDataView = mlContext.Data.LoadFromTextFile<ClassifyModel>(path, hasHeader: true, separatorChar: ',', allowSparse: false);
            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", inputColumnName: nameof(ClassifyModel.Tag))
                            .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "TextFeaturized", inputColumnName: nameof(ClassifyModel.Text)))
                            .Append(mlContext.Transforms.Concatenate(outputColumnName: "Features", "TextFeaturized"))
                            .AppendCacheCheckpoint(mlContext);
            IEstimator<ITransformer> trainer = null;
            switch (selectedStrategy)
            {
                case MyTrainerStrategy.SdcaMultiClassTrainer:
                    trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features");
                    break;
                case MyTrainerStrategy.OVAAveragedPerceptronTrainer:
                    {
                        var averagedPerceptronBinaryTrainer = mlContext.BinaryClassification.Trainers.AveragedPerceptron("Label", "Features", numberOfIterations: 10);
                        trainer = mlContext.MulticlassClassification.Trainers.OneVersusAll(averagedPerceptronBinaryTrainer);

                        break;
                    }
                default:
                    break;
            }
            var trainingPipeline = dataProcessPipeline.Append(trainer)
                    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            var trainedModel = trainingPipeline.Fit(trainingDataView);
            mlContext.Model.Save(trainedModel, trainingDataView.Schema, FileUtil.CreateFileSequence(FileUtil.AppendToFileName(path, "model", ".zip")));
        }
    }
}
