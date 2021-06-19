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
        protected virtual void TestHandler()
        {
            writeline(nameof(TestHandler));

            this.WriteParams();
            writeline("ok");
            this.End();
        }
    }
}
