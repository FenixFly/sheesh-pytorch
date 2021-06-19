using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Web;
using CsvHelper;
using Newtonsoft.Json;

namespace CardioSpike
{
    public partial class BaseHandler 
    {
        protected virtual void RGHandler()
        {
            var id = param("id");
            if (id.IsNullOrWhiteSpace()) throw new AccessViolationException("пустой код ритмограмы");
            
            var rhythmogram = GetRhythmogram(id);
            writejson(rhythmogram).End();

            //writeline(nameof(RGHandler));
            //this.WriteParams();
            //writeline("ok");
            //this.End();
        }


        static public rhythmogram GetRhythmogram(string id)
        {
            if (id.Contains(@"..")) throw new Exception("не корректный id ритмограммы");
            if (id.Contains(@"/")) throw new Exception("не корректный id ритмограммы");
            if (id.Contains(@"\")) throw new Exception("не корректный id ритмограммы");

            string filename = id.Replace("@", "/") + ".csv";
            filename = Path.Combine(filestoragepath(), filename);
            FileInfo fi = new FileInfo(filename);
            if (!fi.Exists) throw new AccessViolationException("нет найдена ритмограмма");
            var rhythmogram = GetRhythmogramFromFile(fi);
            return rhythmogram;

        }

        static public rhythmogram GetRhythmogramFromFile(FileInfo fileinfo)
        {
            var csv = File.ReadAllText(fileinfo.FullName);

            var rhythmogram = RhythmogramFromCsv(csv);
            
            rhythmogram.id = $"{fileinfo.Directory.Name}@{fileinfo.Name.Replace(fileinfo.Extension,"")}";
            if (rhythmogram.id.Contains(@"/")) throw new Exception("не удалось сформировать id ритмограммы");
            if (rhythmogram.id.Contains(@"\")) throw new Exception("не удалось сформировать id ритмограммы");
            if (rhythmogram.id.Contains(@"..")) throw new Exception("не удалось сформировать id ритмограммы");

            rhythmogram.timeString = fileinfo.LastWriteTime.ToString("g");
            return rhythmogram;
        }

        static public rhythmogram RhythmogramFromCsv(string csv)
        {
            var rhythmogram = new rhythmogram();

            List<row> rowlist = null;

            using (var reader = new StringReader(csv))
            using (var csvreader = new CsvReader(reader, System.Globalization.CultureInfo.InvariantCulture))
            {
                rowlist = csvreader.GetRecords<row>().ToList();
            }

            dataset dataset = new dataset();
            dataset.setmode(false);
            rhythmogram.datasets.Add(dataset);
            int lastmode = 0;
            int risk = 0;

            foreach (var row in rowlist)
            {
                if (row.y > 0) risk = 1;

                if (lastmode != row.y)
                {
                    dataset = new dataset();
                    dataset.setmode(row.y > 0);
                    rhythmogram.datasets.Add(dataset);
                }
                dataset.data.Add(new data() { x = row.time, y = row.x });
                lastmode = row.y;
            }

            rhythmogram.risk = risk;

            return rhythmogram;
        }

        static public string RhythmogramCsvToJson(string rid, string csv)
        {
            var rhythmogram = RhythmogramFromCsv(csv);
            rhythmogram.id = rid;

            //json.datasets.data = csvreader.GetRecords<data>().ToList();

            string jsontext = JsonConvert.SerializeObject(rhythmogram, Formatting.Indented);
            return jsontext;
        }

    }
}
