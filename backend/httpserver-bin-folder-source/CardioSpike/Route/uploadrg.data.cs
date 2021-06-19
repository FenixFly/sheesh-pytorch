using System;
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

        public class rhythmogram
        {
            public string id { get; set; }
            public string timeString { get; set; } = DateTime.Now.ToString("g");
            public int risk { get; set; }
            public string cardiogramId { get; set; }

            public List<dataset> datasets { get; set; } = new List<dataset>();
        }

        public class dataset
        {
            public string borderColor { get; set; } 
            public string backgroundColor { get; set; } 
            public bool fill { get; set; } = true;
            public int lineTension { get; set; } = 0;

            internal bool mode = false; 
            public void setmode(bool mode)
            {
                this.mode = mode;
                if (mode == true)
                {
                    borderColor = "#FF0000";
                    backgroundColor = "#FF0000";
                    fill = true;
                }
                else
                {
                    borderColor = "#5AAFC7";
                    backgroundColor = "#5AAFC7";
                    fill = false;
                }
            }

            public List<data> data { get; set; } = new List<data>();
        }
        public class data
        {            public int x { get; set; }
            public int y { get; set; }
        }

        public class row
        {
            public int time { get; set; }
            public int x { get; set; }
            public int y { get; set; }
        }


    }
}
