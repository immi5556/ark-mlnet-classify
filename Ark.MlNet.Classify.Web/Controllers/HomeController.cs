using ark.net.util;
using Ark.MlNet.Classify.Web.Models;
using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
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

        [HttpPost]
        [Route("ark/class/{model_name}")]
        public dynamic Index1([FromBody] List<ClassifyModel> models, [FromRoute] string model_name)
        {
            StringBuilder content = new StringBuilder();
            Dictionary<string, string> dics = new Dictionary<string, string>();
            (models ?? new List<ClassifyModel>()).ForEach(model =>
            {
                content.AppendLine($"{model.Text},{model.Tag}");
                if (!dics.ContainsKey(model.Tag)) dics.Add(model.Tag, model.Tag);
            });
            var path = $"./Data/{model_name}/{ark.net.util.DateUtil.CurrentTimeStamp()}.csv";
            FileUtil.Save(path, content.ToString());
            FileUtil.Save(FileUtil.AppendToFileName(path, "tags", ".json"), System.Text.Json.JsonSerializer.Serialize(new { tags = dics.Keys.ToList() }));
            ArkClassifier.BuildAndTrainModel(MyTrainerStrategy.OVAAveragedPerceptronTrainer, path);
            return new
            {
                message = "trained successful."
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