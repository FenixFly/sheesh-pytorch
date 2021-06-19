using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CardioSpike
{
    public partial class TestHandler : BaseHandler
    {
        protected void Page_Load(object sender, EventArgs e)
        {
            //https://cardiospike.ip3.ru/test.aspx
            this.TestHandler();
        }

    }
}
