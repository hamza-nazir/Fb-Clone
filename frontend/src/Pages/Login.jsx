import React from 'react'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../Hook/Axios'
import { Link } from 'react-router-dom'
import '../css/login.css'
const Login = () => {
    const navigate=useNavigate();
    const [incorrentPass,setIncorrectPass]=useState(false);
        const [loading,setLoading]=useState(false);
    const [formData,setFormData]=useState({email:'',password:''})
    const handleInput=(e)=>{
        setFormData((curr)=>{
            return{
                ...curr,
                [e.target.name]:e.target.value
            }
        })
    }
    const submit= (e)=>{
        setLoading(true)
        e.preventDefault();
       api.post('/login',formData)
       .then((res)=>{
        // console.log(res.data.user);
        navigate('/')
       })
       .catch((err)=>{
        console.log(err.message);
        console.log("INVAlif");
        setIncorrectPass(true)
       })
       .finally(()=>{
        setLoading(false)
       })
               
    }
    return (
        <div id='container-login' className='login container-fluid' style={{backgroundColor:'#F2F4F7',height:'100vh'}}>
            <div className="row p-5">
                <div id='left-login-container' className="col-7 p-5 mt-5">
                    <h1 style={{fontSize:'80px',color:'#0866FF'}}>facebook</h1>
                    <p style={{fontSize:'30px'}}>Facebook helps you connect and share with the people in your life.</p>
                </div>
                <div className="col-5 p-5" id='right-login-container'>
                   <form onSubmit={submit} className='form-control mx-auto p-4' style={{ boxShadow: '0.2px 0.2px 10px #DDDEE1' }}>
                        <input type="text" className='form-control mt-3' placeholder='Enter email' name='email' onChange={(e)=>{handleInput(e);setIncorrectPass(false)}} />
                        <input type="text" className='form-control mt-3' autoComplete='off' placeholder='Enter password' name='password' onChange={(e)=>{handleInput(e);setIncorrectPass(false)}}/>
                        {incorrentPass&&(<p className='mt-2 text-muted'>Incorrent Email or Password</p>)}
                         <div className='d-flex justify-content-center'>
                        <button disabled={!formData.email||!formData.password} className='mt-3 btn btn-primary' style={{width:'100%'}}>
                            {
                                loading?(<span>Logging in...</span>):(<span>Login</span>)
                            }
                        </button>
                          </div>
                        <p className='text-center mt-3'>Forgotten password?</p>
                        <hr />
                        <div className='d-flex justify-content-center'>
                         <Link to={'/signup'}  style={{backgroundColor:'#36A420',border:'none',padding:'10px',color:'white',borderRadius:'7px',textAlign:'center',textDecoration:'none'}}>Create new account</Link>
                      </div>
                    </form>
                </div>
            </div>
        </div>
    )
}

export default Login
