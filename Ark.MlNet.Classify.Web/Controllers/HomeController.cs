using ark.net.util;
using Ark.MlNet.Classify.Web.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using System.Diagnostics;
using System.IO;
using System.Text;

namespace Ark.MlNet.Classify.Web.Controllers
{
    public class HomeController : Controller
    {
        private readonly ILogger<HomeController> _logger;

        public HomeController(ILogger<HomeController> logger)
        {
            _logger = logger;
        }

        public IActionResult Index()
        {
            return View();
        }

        public IActionResult Train()
        {
            Dictionary<string, dynamic> dics = new Dictionary<string, dynamic>();
            foreach (var v in Directory.EnumerateDirectories("./Data"))
            {
                if (string.IsNullOrEmpty(v)) continue;
                var key = System.IO.Path.GetFileName(v);
                if (dics.ContainsKey(key)) continue; //not possible
                DirectoryInfo dir = new DirectoryInfo(v);
                FileInfo? rootfile = dir.GetFiles("*.csv").OrderByDescending(p => p.CreationTimeUtc).FirstOrDefault();
                if (rootfile == null) continue; //worst case
                var modelfile = FileUtil.AppendToFileName(rootfile.FullName, "model", ".zip");
                if (!System.IO.File.Exists(modelfile)) continue; //if this happens -> very bad in persistence process flow
                var tagfile = FileUtil.AppendToFileName(rootfile.FullName, "tags", ".json");
                var tag_content = "N/A";
                if (System.IO.File.Exists(tagfile)) tag_content = System.IO.File.ReadAllText(tagfile);
                dics.Add(key, new
                {
                    model_name = key,
                    model_path = $"./Data/{key}/{System.IO.Path.GetFileName(modelfile)}",
                    tag = tag_content,
                    tag_path = $"./Data/{key}/{System.IO.Path.GetFileName(tagfile)}",
                });
            }
            ViewBag.dics = dics;
            return View();
        }

        [HttpPost]
        [Route("ark/class/train/{alg}/{model_name}")]
        public dynamic Index1([FromBody] List<ClassifyModel> models, [FromRoute] string alg, [FromRoute] string model_name)
        {
            StringBuilder content = new StringBuilder();
            Dictionary<string, string> dics = new Dictionary<string, string>();
            (models ?? new List<ClassifyModel>()).ForEach(model =>
            {
                content.AppendLine($"{model.Text},{model.Tag}");
                if (!dics.ContainsKey(model.Tag)) dics.Add(model.Tag, model.Tag);
            });
            MyTrainerStrategy algorithm = (alg ?? "ova") == "ova" ? MyTrainerStrategy.OVAAveragedPerceptronTrainer : MyTrainerStrategy.SdcaMultiClassTrainer;
            var path = $"./Data/{model_name}/{ark.net.util.DateUtil.CurrentTimeStamp()}.csv";
            System.IO.File.WriteAllText(path, content.ToString());
            System.IO.File.WriteAllText(FileUtil.AppendToFileName(path, "tags", ".json"), System.Text.Json.JsonSerializer.Serialize(new { tags = dics.Keys.ToList(), alg = alg, algorithm = algorithm }));
            ArkClassifier.BuildAndTrainModel(algorithm, path);
            return new
            {
                message = "trained successful."
            };
        }
        [HttpPost]
        [Route("ark/class/predict")]
        public dynamic Index2([FromForm] string model_path, [FromForm] string text)
        {
            var prediction = ArkClassifier.Predict(model_path, text);
            return new
            {
                prediction = prediction.tag,
                score = prediction.score
            };
        }

        public IActionResult Privacy()
        {
            return View();
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}