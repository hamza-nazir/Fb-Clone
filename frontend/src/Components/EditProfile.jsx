import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import IconButton from '@mui/material/IconButton';
import { Context } from '../Hook/Context';
import { useContext, useState,useEffect } from 'react';

import api from '../Hook/Axios'
import CloseIcon from '@mui/icons-material/Close';
const EditProfile = () => {
    const {edit,setEdit,userDetails,setUserDetails}=useContext(Context);
    const [tempProfile,setTempProfile]=useState(null)
    const [tempCover,setTempCover]=useState(null)

  const [formData, setFormData] = useState({
  profileImg: null,
  coverImg: null,
});

useEffect(() => {
  if (userDetails) {
    setFormData({
      profileImg: userDetails.profileImg || null,
      coverImg: userDetails.coverImg || null,
    });
  }
}, [userDetails]);
 


    function submitFun(){
      
        const form_Data=new FormData();
      form_Data.append("bio", formData.bio);
     form_Data.append("profileImg", formData.profileImg); 
      form_Data.append("coverImg", formData.coverImg);
      api.post('/edit-user',form_Data)

      .then((res)=>{
        setUserDetails(res.data.user);
      })
      .catch(((err)=>{
        console.log("ERROR");
      }))
     
    }
    return (
     
            <Dialog open={edit} maxWidth="sm" fullWidth>
           
                  <DialogTitle>
                    <div>
                    <h5 className='text-center' onClick={()=>{
                      console.log("Form-Data",formData)
                   console.log(tempProfile,tempCover);
                     

                      }}>Edit Profile</h5>
                     </div>
                    <IconButton onClick={()=>{
                      setEdit(false);
                      setTempProfile(null);
                      setTempCover(null);
                      setFormData(()=>({profileImg:userDetails?.profileImg||null,coverImg:userDetails?.coverImg||null})); 
                    }} sx={{position: 'absolute',right: 8,top: 8}}>
                      <CloseIcon />
                    </IconButton>
                  </DialogTitle>
                  <DialogContent dividers>

  <div>

  

    <h1>Profile</h1>
  <input onChange={(e)=>{

    setFormData((prev)=>({...prev,profileImg:e.target.files[0]}));

    setTempProfile(URL.createObjectURL(e.target.files[0]));

    }}  type="file" name="" id='profileImg' style={{display:'none'}} />
    <label className='border' htmlFor="profileImg">Select</label>
<br />
{
 userDetails?.profileImg!=null&&tempProfile==null?(
<img style={{width:'150px'}} src={userDetails?.profileImg} />
):tempProfile!=null?(
  <img src={tempProfile} style={{width:'150px'}} />
):(
  <img style={{width:'150px'}} src='/profile.png' />

  )
}

</div>










  <div >
<h1>Cover</h1>
  <input onChange={(e)=>{
    setFormData((prev)=>({...prev,coverImg:e.target.files[0]}))
    setTempCover(URL.createObjectURL(e.target.files[0]))
}}  type="file"  id="coverImg" style={{display:'none'}} />
<label htmlFor="coverImg" className='border'>Select</label>
<br />
  {userDetails?.coverImg!=null&&tempCover==null?(
  <img style={{width:'150px'}} src={userDetails?.coverImg} />
  ):tempCover!=null?(
      <img style={{width:'150px'}} src={tempCover} />
  ):(
  <img style={{width:'150px'}} src='/cover.png' />

  )
}

</div>


<div>
<button className='mx-auto d-flex mt-2 btn btn-primary' onClick={()=>submitFun()}>Submit</button>

</div>
<div>

</div>
                  </DialogContent>
                  </Dialog>
      
    )
}

export default EditProfile
