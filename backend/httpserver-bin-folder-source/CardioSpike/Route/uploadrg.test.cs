using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Web;
using CsvHelper;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace CardioSpike
{
    public partial class BaseHandler 
    {
        static public void testPostToEvgenii()
        {
            var in1 = @"
                        time,x
                        800,0
                        600,0
                       ".NormalizeCsv();
            var out1 = @"
                        time,x,y
                        800;0;0
                        600;0;0
                        ".NormalizeCsv();

            var result = PostToEvgenii(in1);

            if (in1.IsNullOrWhiteSpace()) throw new NotImplementedException();
            if (out1.IsNullOrWhiteSpace()) throw new NotImplementedException();
            if (result != out1) throw new NotImplementedException();

        }


        static public void testEvgeniiCsvToJson()
        {

            var in1 = @"
                        time,x
                        600,0
                        800,0
                        900,0

                       ".NormalizeCsv();
            var out1 = @"
                        time,x,y
                        600;0;0
                        800;0;0
                        900;0;0
                        ".NormalizeCsv();

            
            var jsontext = RhythmogramCsvToJson("111", out1);


            //var json = new rgjson();

            //using (var reader = new StringReader(out1))
            //using (var csv = new CsvReader(reader, System.Globalization.CultureInfo.InvariantCulture))
            //{
            //    json.datasets.data = csv.GetRecords<time_x_y>().ToList();
            //}

            //string jsontext = JsonConvert.SerializeObject(json, Formatting.Indented);

            //JArray array = new JArray();
            //array.Add("Manual text");
            //array.Add(new DateTime(2000, 5, 23));
            //JObject o = new JObject();
            //o["MyArray"] = array;



            //var result = UploadRG("1",in1);

            //if (in1.IsNullOrWhiteSpace()) throw new NotImplementedException();
            //if (out1.IsNullOrWhiteSpace()) throw new NotImplementedException();

            //if (result != out1) throw new NotImplementedException();

        }

    }
}
