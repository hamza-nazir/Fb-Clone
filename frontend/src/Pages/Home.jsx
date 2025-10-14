import {Context} from '../Hook/Context'
import { useContext } from "react";

import Explore from '../Components/Explore';
import Navbar from '../Components/Navbar';
import Contacts from '../Components/Contacts';
import Watch from './Watch';
import Feed from './Feed';
import Market from './Market'
import Groups from './Groups'
import Games from './Games'



const Home = () => {

    const {option}=useContext(Context);


  return (
    <>
   <Navbar/>
    <div className='container-fluid' id='home-main-div' style={{ marginTop: '50px' }}>
      <div className="row" style={{ height: '90vh' }}>
        <div className="col-3" id="explore-main-div"  style={{ backgroundColor: '#F2F4F7' }}>
          <Explore  />
        </div>

        <div className="col-6" id="feed-main-div" style={{backgroundColor:'#F2F4F7' }}>
      
{option=='feed'&&( <Feed/>)}
{option=='watch'&&( <Watch/>)}
{option=='market'&&( <Market/>)}
{option=='groups'&&( <Groups/>)}
{option=='games'&&( <Games/>)}


        </div>

        <div className="col-3" id="contact-main-div" style={{ backgroundColor: '#F2F4F7' }}>
          <Contacts />
        </div>
      </div>
    </div>
     </>
  );
};

export default Home;
