"""CPU analytical tests: Jacobian, time direction, prefix projection, transaction isolation."""
import pytest
import torch
from g05.models.g05.helpers.rtc import RTCContext,current,guided_velocity,project_prefix,use,weights


def test_full_jacobian_is_used():
    x=torch.full((1,4,2),.3);t=torch.tensor([.5]);prefix=torch.ones((1,4,2))
    ctx=RTCContext(prefix,2,2,5)
    # v=0.4*x => endpoint=0.8*x, J=0.8*I. Missing the model Jacobian gives a different result.
    result=guided_velocity(x,t,lambda a:.4*a,ctx,True)
    error=(prefix-.8*x)*weights(ctx,4,x.device,x.dtype)
    expected=.4*x-2*.8*error
    torch.testing.assert_close(result,expected)


def test_time_conventions_are_equivalent():
    x=torch.randn(1,8,3);t=torch.tensor([.7]);ctx=RTCContext(torch.randn(1,5,3),2,3)
    pi=guided_velocity(x,t,lambda a:a.square()*.1,ctx,True)
    gal=guided_velocity(x,1-t,lambda a:-a.square()*.1,ctx,False)
    torch.testing.assert_close(pi,-gal)


def test_projection_follows_noise_then_exact_prefix():
    x=torch.randn(1,8,2);noise=x.clone();ctx=RTCContext(torch.ones(1,4,2),3)
    torch.testing.assert_close(project_prefix(x,noise,torch.ones(1),ctx,True),x)
    final=project_prefix(x,noise,torch.zeros(1),ctx,True)
    torch.testing.assert_close(final[:,:3],ctx.prefix[:,:3])
    torch.testing.assert_close(final[:,3:],x[:,3:])


def test_context_clears_on_failure():
    assert current() is None
    with pytest.raises(RuntimeError):
        with use(RTCContext(torch.ones(1,3,2),1)):
            assert current() is not None
            raise RuntimeError('injected failure')
    assert current() is None


def test_padded_dimensions_have_no_guidance():
    x=torch.zeros(1,4,3);mask=torch.tensor([[[False,False,True]]]);ctx=RTCContext(torch.ones(1,3,3),2)
    v=guided_velocity(x,torch.tensor([.5]),lambda a:a*.1,ctx,True,pad_mask=mask)
    torch.testing.assert_close(v[:,:,2],torch.zeros_like(v[:,:,2]))


def test_invalid_delay_is_rejected():
    with pytest.raises(ValueError):
        RTCContext(torch.ones(1,3,2),4)


def test_known_prefix_normalization_is_invertible_beyond_training_clip():
    from g05.utils.data.normalizer import SingleFieldLinearNormalizer
    norm=SingleFieldLinearNormalizer({'mean':torch.zeros(2),'std':torch.ones(2)},mode='z-score')
    x=torch.tensor([[8.,-9.]])
    assert norm.forward(x).abs().max()==5
    torch.testing.assert_close(norm.backward(norm.forward(x,clip=False)),x)
