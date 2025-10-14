import { useContext } from 'react';
import {Context} from '../Hook/Context'
import '../css/Navbar.css'
import useAuth from "../Hook/useAuth";
import { Link } from "react-router-dom";

const Navbar = () => {
    const currUser=useAuth(); 
    const {option,setOption,setOpen,setLogout,setRequest}=useContext(Context);
       
    return (
        <div className='container-fluid'>
            <div className="row" id='nav' style={{height:'50px'}}>
                <div className="col" id='nav-col-1' >
                  <Link to={'/'}>  <span><i  className="fa-brands fa-facebook-f" id='fb-logo' style={{fontSize:'30px',color:'white'}}></i></span></Link>
                    <input onClick={()=>setOpen(true)}  autoComplete='off' readOnly  type="search" id='nav-search' placeholder='Search Facebook' style={{borderRadius:'18px', border:'none' ,padding:'7px',backgroundColor:'#F0F2F5' }}/>
                </div>
                <div id='nav-col-2' className="col">
                    <i onClick={()=>setOption('feed')} style={option=='feed'?{color:'#0866FF',borderBottom:'2px solid #0866FF',paddingBottom:'4px',cursor:'pointer'}:{cursor:'pointer'}}  className="fa-regular fa-house"></i>
                    <i onClick={()=>setOption('watch')} style={option=='watch'?{color:'#0866FF',borderBottom:'2px solid #0866FF',paddingBottom:'4px',cursor:'pointer'}:{cursor:'pointer'}} className="fa-solid fa-tv"></i>
                    <i onClick={()=>setOption('market')} style={option=='market'?{color:'#0866FF',borderBottom:'2px solid #0866FF',paddingBottom:'4px',cursor:'pointer'}:{cursor:'pointer'}} className="fa-solid fa-store"></i>
                    <i onClick={()=>setOption('groups')} style={option=='groups'?{color:'#0866FF',borderBottom:'2px solid #0866FF',paddingBottom:'4px',cursor:'pointer'}:{cursor:'pointer'}} className="fa-solid fa-people-group"></i>
                    <i onClick={()=>setOption('games')} style={option=='games'?{color:'#0866FF',borderBottom:'2px solid #0866FF',paddingBottom:'4px',cursor:'pointer'}:{cursor:'pointer'}}  className="fa-solid fa-gamepad"></i>
                </div>
                <div id='nav-col-3'  className="col">
       
                    <i  className="nav-right fa-solid fa-bars ms-3"></i>
                    <i style={{cursor:'pointer'}} onClick={()=>setRequest(true)} className="nav-right fa-duotone fa-regular fa-bell ms-3"></i>
                    <i className="nav-right fa-regular fa-message ms-3 me-2"></i>
                    {
                        currUser?.profileImg!=null?(
                    <img className='me-4 ' src={currUser?.profileImg}  style={{cursor:'pointer',height:'35px',width:'35px',borderRadius:'50%',border:'none',objectFit:'cover'}} onClick={()=>setLogout(true)}/>

                        ):(
                    <img className='me-3' src="/profile.png" id='nav-img' style={{cursor:'pointer',border:'none',boxShadow:'1px 1px 5px #cacbceff'}} onClick={()=>setLogout(true)}/>

                        )
                    }
                </div>

            </div>



        </div>
    )
}

export default Navbar
