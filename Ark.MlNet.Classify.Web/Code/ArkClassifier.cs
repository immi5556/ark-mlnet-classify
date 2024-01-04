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
    public class ClassifyPrediction
    {
        [ColumnName("PredictedLabel")]
        public string Classification;

        public float[] Score;
    }
    public enum MyTrainerStrategy : int
    {
        SdcaMultiClassTrainer = 1,
        OVAAveragedPerceptronTrainer = 2
    };
    public class ArkClassifier
    {
        public static (string tag, float? score) Predict(string mode_path, string text)
        {
            var mlContext = new MLContext(seed: 1);
            DataViewSchema modelSchema;
            ITransformer trainedModel = mlContext.Model.Load(mode_path, out modelSchema);
            var issue = new ClassifyModel() { Text = text };
            var predEngine = mlContext.Model.CreatePredictionEngine<ClassifyModel, ClassifyPrediction>(trainedModel);
            var prediction = predEngine.Predict(issue);
            return (prediction.Classification, prediction.Score?[0]);
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
            //mlContext.Model.Save(trainedModel, trainingDataView.Schema, FileUtil.CreateFileSequence(FileUtil.AppendToFileName(path, "model", ".zip")));
            mlContext.Model.Save(trainedModel, trainingDataView.Schema, FileUtil.AppendToFileName(path, "model", ".zip"));
        }
    }
}
