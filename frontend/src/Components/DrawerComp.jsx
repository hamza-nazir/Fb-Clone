import Drawer from '@mui/material/Drawer';
import { useContext, useState } from "react";
import { Context } from "../Hook/Context";
import IconButton from '@mui/material/IconButton';
import {Link} from 'react-router-dom'
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import api from '../Hook/Axios'
const DrawerComp = () => {
      const {open,setOpen}=useContext(Context);
      const [searchUser,setSearchUser]=useState("");
      const [filterUser,setFilterUser]=useState([])

      const submitFun=()=>{
        api.post('/get-user',{searchUser})
        .then((res)=>{
          setFilterUser(res.data.user);
         })
      }
      // console.log(filterUser);
    return (
        
    <Drawer open={open} anchor="left" variant="persistent"  sx={{'.MuiDrawer-paper': { width: 319,height:300, overflowY:'scroll',   boxShadow: '0px 4px 5px -2px #96989bff',borderBottomRightRadius:'10px'  }}}>
    <div className="row pt-2">
     <div className="col" id='nav-col-1' >
      <span>
        <IconButton  onClick={()=>setOpen(false)} className='mt-1 me-2'   >
        <ArrowBackIcon  />
      </IconButton>
      </span>
       <input onChange={(e)=>setSearchUser(e.target.value)} autoComplete='off' className='me-2'  type="search" id='nav-search' placeholder='Search Facebook' style={{borderRadius:'18px', border:'none' ,padding:'7px',width:'180px',backgroundColor:'#F0F2F5' }}/>
      <button disabled={!searchUser} onClick={submitFun} className='btn btn-primary pe-2'>Submit</button>
      </div>
     </div>
     <hr />
     <div >
      
             {
        filterUser?.map((item,index) => (
          <Link to={`/user/${item._id}`} onClick={()=>setOpen(false)} style={{textDecoration:'none',color:'black'}} key={index} className="row">
            <div className="col-3">


            {item?.profileImg!=null?(
             <img className='ms-3' style={{borderRadius:'50%',border:'1px solid black'}} src={item?.profileImg} width={40} />

            ):(
             <img className='ms-3' style={{borderRadius:'50%',border:'1px solid black'}} src="/profile.png" width={40} />

            )
          }

            </div>
            <div className="col mt-2">
                  <b style={{textDecoration:'none',color:'black'}} >{item.name}</b>
            </div>
            <hr style={{width:'90%',color:'grey'}} className='mx-auto mt-2' />
          </Link>
    
        ))
      }
          
     
     

     </div>
    </Drawer>
     
)}

export default DrawerComp
