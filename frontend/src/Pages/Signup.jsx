import { useEffect, useState } from "react"
import api from '../Hook/Axios'
import { useNavigate } from "react-router-dom"
const Signup = () => {
    const navigate=useNavigate()
    const [formData,setFormData]=useState({name:'',surname:'',email:'',password:'',confirmPassword:'',gender:'male' })
    const [passSame,setPassSame]=useState(false);
    const [loading,setLoading]=useState(false);
    const [emailExist,setEmailExist]=useState(false)
    const handleChange=(e)=>{
        setFormData((curr)=>{
            return {
                ...curr,
                [e.target.name]:e.target.value,
            }
        })
    }
    const submit=async (e)=>{
        setLoading(true)
        e.preventDefault();
       await api.post('/signup',formData)
       .then((res)=>{
        if(res.data.success===true){
            console.log(res.data);
            setEmailExist(false);
            // setLoading(false);
            navigate('/')
        }else{
            console.log(res.data);
            setEmailExist(true);
            // setLoading(true);
        }
       })
       .finally(()=>{
          setLoading(false);
       })
       
    }

    useEffect(()=>{
        if(formData.confirmPassword.length>0&&formData.password.length>0&&formData.password==formData.confirmPassword){
            setPassSame(true);
            console.log("Same");
        }else{
              setPassSame(false);
                 console.log("Not Same");
        }
    },[formData.confirmPassword,formData.password])
    return (
      <div className='container-fluid'>
        <h1 className='text-center mt-4 mb-5' style={{fontSize:'80px',color:'blue'}}>facebook</h1>
        <form className='form-control w-50 mx-auto mb-5'>
        <h2 className='text-center'>Create a new account</h2>
        <p className='text-center text-muted'>It's quick and easy.</p>
        <hr />
        <div className="row mb-3">
            <div className="col-lg-12 col-sm-12 mb-3">
                <input className='form-control' type="text" placeholder='Full Name' onChange={handleChange} name='name' required />
            </div>
         

        </div>
        <div className="row mb-3">
           
            <div className="col ">
        
            <div style={{width:'100%',display:'flex',justifyContent:'space-between',padding:'10px',border:'1px solid #CCD0D5',backgroundColor:'white',borderRadius:'7px'}}> <label htmlFor='female'>Female</label> <input type="radio" required name="gender" onChange={()=>setFormData(prev=>( {...prev,gender: 'female'}))} id='female'  /> </div>
            </div>
            <div className="col">
            <div style={{width:'100%',display:'flex',justifyContent:'space-between',padding:'10px',border:'1px solid #CCD0D5',backgroundColor:'white',borderRadius:'7px'}}> <label htmlFor="male">Male</label> <input type="radio" name="gender"  onChange={()=>setFormData(prev=>({...prev, gender: 'male'}))}  id='male' /> </div>
            </div>
            <div className="col">
             <div style={{width:'100%',display:'flex',justifyContent:'space-between',padding:'10px',border:'1px solid #CCD0D5',backgroundColor:'white',borderRadius:'7px'}}> <label htmlFor="custom">Custom</label> <input type="radio" name="gender"   onChange={()=>setFormData(prev=>({...prev, gender: 'custom'}))} id='custom' /> </div>
            </div>
        </div>

        <div className="row mb-3">
            <div className="col">
                <input type="email" className='form-control' placeholder='Email Address'  onChange={(e)=>{handleChange(e);setEmailExist(false)}} required name='email' />
            {emailExist&&(<p className="mt-1 ms-2 text-muted"> Email already Exist</p>)}
            </div>
        </div>

          <div className="row mb-3">
            <div className="col">
                <input type="text" className='form-control' placeholder='Enter Password'  onChange={handleChange} required name='password' />
            </div>
        </div>

        <div className="row mb-3">
            <div className="col">
                <input type="text" className='form-control' placeholder='Confirm Password'  onChange={handleChange} required name='confirmPassword' />
            </div>
            {
                !passSame&&formData.confirmPassword&&(<p className="text-muted mt-2">Password not Match</p>)
            }
        </div>

        <div className="row">
            <div className="col">
                <p className='text-muted'>
                    People who use our service may have uploaded your contact information to Facebook. Learn more.
                </p>
                <p className='text-muted'>
                    By clicking Sign Up, you agree to our Terms, Privacy Policy and Cookies Policy. You may receive SMS notifications from us and can opt out at any time.
                </p>
            </div>
        </div>
        <div className="row">
            <div className="col d-flex justify-content-center">
                <button style={{width:'200px'}} disabled={!passSame||!formData.name||!formData.email||!formData.password||!formData.confirmPassword}  onClick={submit}  className='btn btn-primary  p-2'>{
                    loading?(<span>Signing up...</span>):(<span>Signup</span>)
                }</button>
            </div>
        </div>
        </form>
      </div>
    )
}

export default Signup
