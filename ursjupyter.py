import sys
import requests
import getpass
from IPython.display import display, clear_output, HTML

# if Python2, prompt with raw_input instead of input
if sys.version_info.major==2:
    input = raw_input


class EarthdataLogin(requests.Session):
    """
    Prompt user for Earthdata credentials repeatedly until auth success. Source:
    https://wiki.earthdata.nasa.gov/display/EL/How+To+Access+Data+With+Python
    """

    AUTH_HOST = "urs.earthdata.nasa.gov"              # urs login url
    
    ERROR = "Login failed ({0}). Retry or register."  # failure message
    
    TEST = ("https://daac.ornl.gov/daacdata/")        # test authentication; 
    
    REGISTER = HTML(                                  # registration prompt
        "<p style='font-weight:bold'><a href=https://urs.earth"
        "data.nasa.gov/users/new target='_blank'>Click here to"
        " register a NASA Earthdata account.</a></p>")
                              
    
    def __init__(self):
        fails = 0
        
        while True:
            display(self.REGISTER)                     # register prompt
            username = input("Username: ")             # username prompt 
            password = getpass.getpass("Password: ")   # secure pw prompt
            
            if sys.version_info.major==2:              # init requests session
                super(EarthdataLogin, self).__init__() # for Python 2
            else:
                super().__init__()                     # for Python 3
            self.auth = (username, password)           # add username,password
            
            try:                                     
                response = self.get(self.TEST)         # try to grab TEST
                response.raise_for_status()            # raise for status>400
                clear_output()                         # clear output
                display("Login successful. Download with: session.get(url)")
                break

            except requests.exceptions.HTTPError as e:
                clear_output()                         # clear cell output
                fails += 1                             # +1 fail counter
                display(self.ERROR.format(str(fails))) # print failure msg

    
    def rebuild_auth(self, prepared_request, response):
        """
        Overrides from the library to keep headers when redirected to or 
        from the NASA auth host.
        """
        
        headers = prepared_request.headers
        url = prepared_request.url
 
        if 'Authorization' in headers:
            original_parsed = requests.utils.urlparse(response.request.url)
            redirect_parsed = requests.utils.urlparse(url)
            
            if (original_parsed.hostname != redirect_parsed.hostname) and \
                    redirect_parsed.hostname != self.AUTH_HOST and \
                    original_parsed.hostname != self.AUTH_HOST:
                del headers['Authorization']
                   
        self.auth = None   # drop username/password attributes
        return             # return requests.Session


### RUN ON IMPORT
session = EarthdataLogin()
