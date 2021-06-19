using CsvHelper;
using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Web;

namespace CardioSpike
{
    public partial class BaseHandler 
    {
        public class patient
        {
            public string id { get; set; }
            public string name { get; set; }
            public int age { get; set; } = 30;

            public string lastRgTimeString { get; set; }
            public int risk { get; set; }
            public bool doctor { get; set; }
            public string photo { get; set; } = @"https://cdn.quasar.dev/img/avatar.png";
            public List<rhythmogram> rhythmograms { get; set; } = new List<rhythmogram>();
        }

        //public class patientlist : List<patient> { }

        protected virtual void PatientHandler()
        {
            var id = param("id");
            if (id.IsNullOrWhiteSpace()) throw new AccessViolationException("не верный ключ пациента");

            var patient = GetPatient(id);
            writejson(patient).End();

            writeline(nameof(PatientsHandler));
            writeline(filestoragepath());
            this.End();
        }

        static public patient GetPatient(string id)
        {
            if (id.IsNullOrWhiteSpace()) throw new AccessViolationException("не верный ключ пациента");
            DirectoryInfo dir = new DirectoryInfo(filestoragepath(id));
            if (!dir.Exists) throw new Exception("нет папки пациента");

            var p = new patient();
            p.id = dir.Name;
            p.name = dir.Name;
            p.lastRgTimeString = dir.LastWriteTime.ToString("g");

            var filelist = dir.EnumerateFiles()
                .Where(f=>f.Name.Contains("result",StringComparison.OrdinalIgnoreCase) && f.Extension.Equals(".csv",StringComparison.OrdinalIgnoreCase))
                .OrderByDescending(f => f.LastWriteTime)
                .ToList();

            int? risk = null;

            foreach (var file in filelist)
            {
                var rg = GetRhythmogramFromFile(file);
                p.rhythmograms.Add(rg);
                if (!risk.HasValue) risk = rg.risk; //берем одну свежуюж

                rg.datasets = null; //это множественный показыватель
            }

            if (risk.HasValue && risk.Value > 0) p.risk = 1;

            return p;
        }



        static public void testGetRgList()
        {
            var list = GetPatientList().OrderByDescending(d => d.lastRgTimeString).ToList();
            if (list.Count < 150) throw new NotImplementedException();

            var s = Converter.tojson(list);
        }
    }
}
